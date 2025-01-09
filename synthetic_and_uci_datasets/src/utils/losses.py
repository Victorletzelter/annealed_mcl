
import torch
from torch.nn.modules.loss import _Loss

class WTALoss(_Loss):
    """Class for WTA loss (and variants).
    """
    __constants__ = ['reduction']

    def __init__(self,
                 reduction='mean',
                 mode = 'wta',
                 distance = 'euclidean-squared',
                 epsilon=0.05,
                 conf_weight = 1,
                 output_dim=None,
                 temperature=None) -> None:
        """Constructor for the WTA loss.

        Args:
            reduction (str, optional): Type of reduction performed. Defaults to 'mean'.
            mode (str, optional): Winner-takes-all variant ('wta', 'wta-relaxed', 'stable_awta') to choose. Defaults to 'wta'.
            distance (str, optional): Underlying distance to use for the WTA computation. Defaults to 'euclidean'.
            epsilon (float, optional): Value of epsilon when applying the wta-relaxed variant. Defaults to 0.05.
            conf_weight (int, optional): Weight of the confidence loss (beta parameter). Defaults to 1.
            output_dim (int, optional): Dimension of the output space. Defaults to None.
            temperature (float, optional): Temperature parameter for the stable_awta variant. Defaults to None.
        """

        super(WTALoss, self).__init__(reduction)

        assert output_dim != None, "The output dimension must be defined"

        self.mode = mode
        self.distance = distance
        self.epsilon = epsilon
        self.conf_weight = conf_weight
        self.output_dim = output_dim
        self.temperature = temperature
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) :
        """Forward pass for the WTA Loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batch,self.num_hypothesis,output_dim]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,Max_sources],[batch,Max_sources,output_dim]

        Returns:
            loss (torch.tensor)
        """

        hyps_pred_stacked, conf_pred_stacked = predictions # Shape [batch,self.num_hypothesis,output_dim], [batch,self.num_hypothesis,1]
        target_position, source_activity_target = targets # Shape [batch,Max_sources,output_dim],[batch,Max_sources,1]

        losses = torch.tensor(0.)
                
        source_activity_target = source_activity_target[:,:].detach()
        target_position =target_position[:,:,:].detach()
        
        loss=self.sampling_conf_loss_ambiguous_gts(hyps_pred_stacked=hyps_pred_stacked, 
                                                        conf_pred_stacked=conf_pred_stacked,
                                                        source_activity_target=source_activity_target, 
                                                        target_position=target_position, 
                                                        mode=self.mode,
                                                        distance=self.distance,
                                                        epsilon=self.epsilon,
                                                        conf_weight=self.conf_weight)
        losses = torch.add(losses,loss)

        return losses
    
    def sampling_conf_loss_ambiguous_gts(self, hyps_pred_stacked, conf_pred_stacked, source_activity_target, target_position, mode, distance, epsilon, conf_weight):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,output_dim). Represents the predicted hypotheses $f_{\theta}^k(x), k \in {1, \dots, n}$.
            source_activity_target (torch.tensor): Input tensor of shape (batch,Max_sources). Useful when multiple targets {y_i} are available for a given input x. When a single target is available, which corresponds to the usual setup, source_activity_target[:,0] is set to 1, and the other values are set to 0.
            conf_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,1). Represents the predicted confidence scores $\gamma_{\theta}^k(x), k \in {1, \dots, n}$.
            target_position (torch.tensor): Input tensor of shape (batch,Max_sources,output_dim). Represents the ground-truth positions of the targets, each of them of shape (output_dim).
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'wta'.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        filling_value = 100000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_pred_stacked.shape[1]
        batch = source_activity_target.shape[0]
        Max_sources = source_activity_target.shape[1]

        #1st padding related to the inactive targets, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0
        mask_inactive_sources = mask_inactive_sources.expand_as(target_position)
        target_position[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,output_dim]
        
        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,output_dim], such that each of the 
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of target_position along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive targets, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  target_position.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_pred_stacked_duplicated = hyps_pred_stacked.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]

        ### Management of the confidence part
        conf_pred_stacked = torch.squeeze(conf_pred_stacked,dim=-1) #(batch,num_hyps), predicted confidence scores for each hypothesis.
        gt_conf_stacked_t = torch.zeros_like(conf_pred_stacked, device=conf_pred_stacked.device) #(batch,num_hyps), will contain the ground-truth of the confidence scores. 
        
        # assert gt_conf_stacked_t.shape == (batch,num_hyps)

        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hyps,output_dim]
            dist_matrix = torch.sqrt(torch.sum(diff, dim=-1))  #Distance matrix [batch,Max_sources,num_hyps]

        elif distance=='euclidean-squared' :
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hyps,2]
            dist_matrix = torch.sum(diff, dim=-1) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 
            
            # We select the best hypothesis for each source
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]

            mask = wta_dist_matrix <= filling_value #We create a mask of shape [batch,Max_sources] for only selecting the active targets, i.e. those which were not filled with fake values. 
            wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active targets.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of active targets for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   

                selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            
        elif mode == 'wta-relaxed':
        
            #We compute the loss for the "best" hypothesis but also for the others with weight epsilon.  
            
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources], idx_selected of shape [batch,Max_sources].
            
            # assert wta_dist_matrix.shape == (batch,Max_sources)
            # assert idx_selected.shape == (batch,Max_sources)
            
            mask = wta_dist_matrix <= filling_value #We create a mask for only selecting the active targets, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_sources] ; we select only the active targets. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of active targets as a sum over the batch for the computation of the mean (below).

            ### Confidence management
            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)
            ###

            if count_non_zeros_1>0 : 
                loss0 = torch.multiply(torch.sum(wta_dist_matrix)/count_non_zeros_1, 1 - epsilon) #Scalar (average with coefficient)
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   

                selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss0 = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)

            #We then the find the other hypothesis, and compute the epsilon weighted loss for them
            
            # At first, we remove hypothesis corresponding to "fake" ground-truth.         
            large_mask = dist_matrix <= filling_value # We remove entries corresponding to "fake"/filled ground truth in the tensor dist_matrix on
            # which the min operator was not already applied. Shape [batch,Max_sources,num_hypothesis]
            dist_matrix = dist_matrix*large_mask # Shape [batch,Max_sources,num_hypothesis].
            
            # We then remove the hypothesis selected above (with minimum dist)
            mask_selected = torch.zeros_like(dist_matrix,dtype=bool) #Shape [batch,Max_sources,num_hypothesis]
            mask_selected.scatter_(2, idx_selected.unsqueeze(-1), 1) # idx_selected new shape: [batch,Max_sources,1]. 
            # The assignement mask_selected[i,j,idx_selected[i,j]]=1 is performed. 
            # Shape of mask_selected: [batch,Max_sources,num_hypothesis]
            
            # assert mask_selected.shape == (batch,Max_sources,num_hyps)
            
            mask_selected = ~mask_selected #Shape [batch,Max_sources,num_hypothesis], we keep only the hypothesis which are not the minimum.
            dist_matrix = dist_matrix * mask_selected #Shape [batch,Max_sources,num_hypothesis]
            
            # Finally, we compute the loss
            count_non_zeros_2 = torch.sum(dist_matrix!=0)

            if count_non_zeros_2 > 0 :
                epsilon_loss = torch.multiply(torch.sum(dist_matrix)/count_non_zeros_2, epsilon) #Scalar for each hyp
            else : 
                epsilon_loss = torch.tensor(0.)
            
            sum_losses = torch.add(sum_losses, epsilon_loss) # Loss for the unselected (i.e., not winners) hypothesis (epsilon weighted)
            sum_losses = torch.add(sum_losses, loss0) # Loss for the selected (i.e., the winners) hypothesis (1-epsilon weighted)
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss) # Loss for the confidence prediction. 

        elif mode == 'stable_awta':

            # We select the best hypothesis for each source
            _, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]

            boltzmann_dist = torch.exp(-dist_matrix/self.temperature) #Shape [batch,Max_sources,num_hyps]
            boltzmann_dist = boltzmann_dist.detach() # Backpropagation is not performed through the Boltzmann distribution
            sums = torch.sum(boltzmann_dist, dim=2, keepdim=True)  # shape [batch,Max_sources,1], sum along the last dimension, keeping dimension
            sums_expanded = sums.expand_as(boltzmann_dist)  # shape [batch,Max_sources,num_hyps], expand the sums to the same shape as the original tensor

            # Create a mask where sums are non-zero
            non_zero_mask_expanded = sums_expanded != 0
            
            # normalize the dist
            boltzmann_dist[non_zero_mask_expanded] = boltzmann_dist[non_zero_mask_expanded]/sums_expanded[non_zero_mask_expanded] #Shape [batch,Max_sources,num_hyps]

            awta_dist_matrix = boltzmann_dist*dist_matrix #Shape [batch,Max_sources,num_hyps]
            awta_dist_matrix = torch.sum(awta_dist_matrix, dim=-1) #Shape [batch,Max_sources]
            mask = (source_activity_target == 1).squeeze(-1) #We create a mask of shape [batch,Max_sources] for only selecting the active targets, i.e. those which were not filled with fake values. 
            awta_dist_matrix = awta_dist_matrix*mask #[batch,Max_sources], we select only the active targets.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of active targets for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(awta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   

                selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)

        return sum_losses
   