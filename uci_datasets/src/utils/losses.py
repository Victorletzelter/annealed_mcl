
import torch
from torch.nn.modules.loss import _Loss

class mhloss(_Loss):
    """Class for multi-hypothesis (i.e., Winner-Takes-Loss variants) losses.
    """

    def __init__(self,            
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'euclidean-squared',
                 epsilon=0.05,
                 single_target_loss=False,
                 output_dim=None) -> None:
        """Constructor for the multi-hypothesis loss.

        Args:
            reduction (str, optional): Type of reduction performed. Defaults to 'mean'.
            mode (str, optional): Winner-takes-all variant ('wta', 'wta-relaxed', 'wta-top-n') to choose. Defaults to 'wta'.
            top_n (int, optional): Value of n when applying the top_n variant. Defaults to 1.
            distance (str, optional): Underlying distance to use for the WTA computation. Defaults to 'euclidean'.
            epsilon (float, optional):  Value of epsilon when applying the wta-relaxed variant. Defaults to 0.05.
            single_target_loss (bool, optional): Whether to perform single target update (used in ensemble_mode). Defaults to False.
        """
        super(mhloss, self).__init__(reduction)

        assert output_dim != None, "The output dimension must be defined"

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.single_target_loss = single_target_loss
        self.output_dim = output_dim
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) :
        """Forward pass for the multi-hypothesis loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxself.num_hypothesisxoutput_dim]
            targets (torch.Tensor,torch.Tensor): Tuple of shape [batch,Max_sources],[batch,Max_sources,output_dim], where Max_sources is the maximum number of targets for each input.

        Returns:
            loss (torch.tensor)
        """

        if len(predictions) >= 1 :
            hyps_pred_stacked = predictions[0]

        # hyps_pred_stacked,_ = predictions #Shape [batchxself.num_hypothesisxoutput_dim]
        target_position, source_activity_target = targets #Shape [batch,Max_sources,output_dim],[batch,Max_sources,1]

        losses = torch.tensor(0.)
                
        source_activity_target = source_activity_target[:,:].detach()
        target_position =target_position[:,:,:].detach()
        
        loss=self.sampling_loss_ambiguous_gts(hyps_pred_stacked=hyps_pred_stacked, 
                                                            source_activity_target=source_activity_target, 
                                                            target_position=target_position, 
                                                            mode=self.mode,
                                                            top_n=self.top_n, 
                                                            distance=self.distance,
                                                            epsilon=self.epsilon,
                                                            single_target_loss=self.single_target_loss)
        losses = torch.add(losses,loss)

        return losses
    
    def sampling_loss_ambiguous_gts(self, hyps_pred_stacked, source_activity_target, target_position, mode='wta',top_n=1, distance='euclidean', epsilon=0.05, single_target_loss = False):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target torch.tensor): Input tensor of shape (batch,Max_sources)
            target_position (torch.tensor): Input tensor of shape (batch,Max_sources,output_dim)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'wta'.
            top_n (int, optional): Top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): Underlying distance to use. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "inactive targets".  
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_pred_stacked.shape[1]
        batch = source_activity_target.shape[0]
        Max_sources = source_activity_target.shape[1]
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0
        mask_inactive_sources = mask_inactive_sources.expand_as(target_position)
        target_position[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,output_dim]
        
        #We can check whether the operation is performed correctly
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,1]!=filling_value).sum(axis=1).all())
        
        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the 
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of target_position along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive sources, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  target_position.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]
        
        assert gts.shape==(batch,Max_sources,num_hyps,self.output_dim)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_pred_stacked_duplicated = hyps_pred_stacked.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]

        assert hyps_pred_stacked_duplicated.shape==(batch,Max_sources,num_hyps,self.output_dim)
        
        eps = 0.001
        
        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hypothesis,output_dim]
            channels_sum = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
            assert channels_sum.shape == (batch,Max_sources,num_hyps)
            assert (channels_sum>=0).all()
            
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_sources,num_hypothesis]
            
            assert dist_matrix.shape == (batch,Max_sources,num_hyps)

        elif distance=='euclidean-squared' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hypothesis,output_dim]
            dist_matrix = torch.sum(diff, dim=3) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 

            if single_target_loss==True:
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources] 
                wta_dist_matrix, idx_source_selected = torch.min(wta_dist_matrix,dim=1) 
                wta_dist_matrix = wta_dist_matrix.unsqueeze(-1) #[batch,1]
                assert wta_dist_matrix.shape == (batch,1)
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,1], we select only the active sources.
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below).

            else : 
                wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources] 
                mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with fake values. 
                wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources. 
                count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff. 
            else :
                loss = torch.tensor(0.)    
            
            sum_losses = torch.add(sum_losses, loss) 
             
        return sum_losses
    
class mhconfloss(_Loss):
    """Class for rMCL loss (and variants).
    """
    __constants__ = ['reduction']

    def __init__(self,
                 reduction='mean',
                 mode = 'wta',
                 top_n = 1,
                 distance = 'euclidean-squared',
                 epsilon=0.05,
                 conf_weight = 1,
                 rejection_method = 'all',
                 number_unconfident = 1,
                 output_dim=None,
                 temperature=None) -> None:
        """Constructor for the rMCL loss.

        Args:
            reduction (str, optional): Type of reduction performed. Defaults to 'mean'.
            mode (str, optional): Winner-takes-all variant ('wta', 'wta-relaxed', 'wta-top-n') to choose. Defaults to 'wta'.
            top_n (int, optional): Value of n when applying the top_n variant. Defaults to 1.
            distance (str, optional): Underlying distance to use for the WTA computation. Defaults to 'euclidean'.
            epsilon (float, optional): Value of epsilon when applying the wta-relaxed variant. Defaults to 0.05.
            conf_weight (int, optional): Weight of the confidence loss (beta parameter). Defaults to 1.
            rejection_method (str, optional): Type of rejection, i.e., update of the negative hypothesis to perform. Defaults to 'uniform_negative'.
            number_unconfident (int, optional): Number of negative hypothesis to update when the rejection method is 'uniform_negative'. Defaults to 1.
        """

        super(mhconfloss, self).__init__(reduction)

        assert output_dim != None, "The output dimension must be defined"

        self.mode = mode
        self.top_n = top_n
        self.distance = distance
        self.epsilon = epsilon
        self.conf_weight = conf_weight
        self.rejection_method = rejection_method
        self.number_unconfident = number_unconfident
        self.output_dim = output_dim
        self.temperature = temperature
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) :
        """Forward pass for the Multi-hypothesis rMCL Loss. 

        Args:
            predictions (torch.Tensor): Tensor of shape [batchxself.num_hypothesisxoutput_dim]
            targets (torch.Tensor,torch.Tensor): #Shape [batch,Max_sources],[batch,Max_sources,output_dim]

        Returns:
            loss (torch.tensor)
        """

        hyps_pred_stacked, conf_pred_stacked = predictions #Shape [batchxself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]
        target_position, source_activity_target = targets #Shape [batch,Max_sources,output_dim],[batch,Max_sources,1]

        losses = torch.tensor(0.)
                
        source_activity_target = source_activity_target[:,:].detach()
        target_position =target_position[:,:,:].detach()
        
        loss=self.sampling_conf_loss_ambiguous_gts(hyps_pred_stacked=hyps_pred_stacked, 
                                                        conf_pred_stacked=conf_pred_stacked,
                                                        source_activity_target=source_activity_target, 
                                                        target_position=target_position, 
                                                        mode=self.mode,
                                                        top_n=self.top_n, 
                                                        distance=self.distance,
                                                        epsilon=self.epsilon,
                                                        conf_weight=self.conf_weight,
                                                        rejection_method=self.rejection_method,
                                                        number_unconfident=self.number_unconfident)
        losses = torch.add(losses,loss)

        return losses
    
    def sampling_conf_loss_ambiguous_gts(self, hyps_pred_stacked, conf_pred_stacked, source_activity_target, target_position, mode='wta',top_n=1, distance='euclidean', epsilon=0.05, conf_weight = 1.,rejection_method='all',number_unconfident = 3):
        """Winner takes all loss computation and its variants.

        Args:
            hyps_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,2)
            source_activity_target torch.tensor): Input tensor of shape (batch,Max_sources)
            conf_pred_stacked (torch.tensor): Input tensor of shape (batch,num_hyps,1)
            target_position (torch.tensor): Input tensor of shape (batch,Max_sources,output_dim)
            mode (str, optional): Variant of the classical WTA chosen. Defaults to 'epe'.
            top_n (int, optional): top_n winner in the Evolving WTA mode. Defaults to 1.
            distance (str, optional): _description_. Defaults to 'euclidean'.

        Returns:
            loss (torch.tensor)
        """
        
        # hyps_pred_stacked of shape [batch,num_hyps,output_dim]
        # source_activity_target of shape [batch,Max_sources]
        # target_position of shape [batch,Max_sources,output_dim]
        
        filling_value = 1000 #Large number (on purpose) ; computational trick to ignore the "fake" ground truths.
        # whenever the sources are not active, as the source_activity is not to be deduced by the model is these settings. 
        num_hyps = hyps_pred_stacked.shape[1]
        batch = source_activity_target.shape[0]
        Max_sources = source_activity_target.shape[1]

        # assert num_hyps > number_unconfident, "The number of hypothesis is too small comparing to the number of unconfident hypothesis selected in the scoring" # We check that the number of hypothesis is higher than the number of "negative" hypothesis sampled. 
        
        #1st padding related to the inactive sources, not considered in the error calculation (with high error values)
        mask_inactive_sources = source_activity_target == 0
        mask_inactive_sources = mask_inactive_sources.expand_as(target_position)
        target_position[mask_inactive_sources] = filling_value #Shape [batch,Max_sources,output_dim]
        
        #We can check whether the operation is performed correctly
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,0]!=filling_value).sum(axis=1).all())
        # assert (source_activity_target.sum(axis=1).all()==(target_position[:,:,1]!=filling_value).sum(axis=1).all())
        
        #The ground truth tensor created is of shape [batch,Max_sources,num_hyps,2], such that each of the 
        # tensors gts[batch,i,num_hypothesis,output_dim] contains duplicates of target_position along the num_hypothesis
        # dimension. Note that for some values of i, gts[batch,i,num_hypothesis,output_dim] may contain inactive sources, and therefore 
        # gts[batch,i,j,2] will be filled with filling_value (defined above) for each j in the hypothesis dimension.
        gts =  target_position.unsqueeze(2).repeat(1,1,num_hyps,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]
        
        # assert gts.shape==(batch,Max_sources,num_hyps,self.output_dim)
        
        #We duplicate the hyps_stacked with a new dimension of shape Max_sources
        hyps_pred_stacked_duplicated = hyps_pred_stacked.unsqueeze(1).repeat(1,Max_sources,1,1) #Shape [batch,Max_sources,num_hypothesis,output_dim]

        # assert hyps_pred_stacked_duplicated.shape==(batch,Max_sources,num_hyps,2)
        
        eps = 0.001

        ### Management of the confidence part
        conf_pred_stacked = torch.squeeze(conf_pred_stacked,dim=-1) #(batch,num_hyps), predicted confidence scores for each hypothesis.
        gt_conf_stacked_t = torch.zeros_like(conf_pred_stacked, device=conf_pred_stacked.device) #(batch,num_hyps), will contain the ground-truth of the confidence scores. 
        
        # assert gt_conf_stacked_t.shape == (batch,num_hyps)

        if distance=='euclidean' :
            #### With euclidean distance
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hyps,output_dim]
            channels_sum = torch.sum(diff, dim=-1) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
            # assert channels_sum.shape == (batch,Max_sources,num_hyps)
            # assert (channels_sum>=0).all()
            
            dist_matrix = torch.sqrt(channels_sum + eps)  #Distance matrix [batch,Max_sources,num_hyps]
            
            # assert dist_matrix.shape == (batch,Max_sources,num_hyps)

        elif distance=='euclidean-squared' :
            diff = torch.square(hyps_pred_stacked_duplicated-gts) #Shape [batch,Max_sources,num_hyps,2]
            dist_matrix = torch.sum(diff, dim=-1) #Sum over the two dimensions (azimuth and elevation here). Shape [batch,Max_sources,num_hypothesis]
            
        sum_losses = torch.tensor(0.)

        if mode == 'wta': 
            
            # We select the best hypothesis for each source
            wta_dist_matrix, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]

            mask = wta_dist_matrix <= filling_value/2 #We create a mask of shape [batch,Max_sources] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            wta_dist_matrix = wta_dist_matrix*mask #[batch,Max_sources], we select only the active sources.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(wta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
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
            
            mask = wta_dist_matrix <= filling_value/2 #We create a mask for only selecting the actives sources, i.e. those which were not filled with
            wta_dist_matrix = wta_dist_matrix*mask #Shape [batch,Max_sources] ; we select only the active sources. 
            count_non_zeros_1 = torch.sum(mask!=0) #We count the number of actives sources as a sum over the batch for the computation of the mean (below).

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
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method =='uniform_negative' :
                    # assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."

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
            
        elif mode == 'wta-top-n' and top_n > 1:
            
            # dist_matrix.shape == (batch,Max_sources,num_hyps)
            # wta_dist_matrix of shape [batch,Max_sources]
            
            dist_matrix = torch.multiply(dist_matrix, -1) # Shape (batch,Max_sources,num_hyps) 
            top_k, indices = torch.topk(input=dist_matrix, k=top_n, dim=-1) #top_k of shape (batch,Max_sources,top_n), indices of shape (batch,Max_sources,top_n) 
            dist_matrix_min = torch.multiply(top_k,-1) 
            
            mask = dist_matrix_min <= filling_value/2 # We create a mask of shape [batch,Max_sources,top_n] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            # assert mask[:,:,0].all() == mask[:,:,-1].all() # This mask related should be constant in the third dimension.
            
            dist_matrix_min = dist_matrix_min*mask # [batch,Max_sources,top_n], we select only the active sources.  
            # assert dist_matrix_min.shape == (batch,Max_sources,top_n)
            
            count_non_zeros = torch.sum(mask[:,:,0]!=0) # We count the number of entries (in the first two dimensions) for which the mask is different from zero. 
            
            ### Confidence management
            # Create tensors to index batch and Max_sources and top-n dimensions. 
            batch_indices = torch.arange(batch)[:, None, None].expand(-1, Max_sources,top_n) # Shape (batch, Max_sources,top_n)
            # We set the confidence of the selected hypothesis
            gt_conf_stacked_t[batch_indices[mask], indices[mask]] = 1 #Shape (batch,num_hyps)
            ###

            #####
            selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores 
            # to be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.
            
            for i in range(top_n):

                # assert count_non_zeros == torch.sum(mask[:,:,i]!=0) # We count the number of entries for which the mask is different from zero. 
            
                if count_non_zeros>0 : 
                    loss  = torch.multiply(torch.sum(dist_matrix_min[:, :, i])/count_non_zeros, 1.0)

                else :
                    loss = torch.tensor(0.) 

                sum_losses = torch.add(sum_losses, loss/top_n)

            if count_non_zeros>0 : 
              
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                ### Uncomment the following lines to check that the selected_confidence_mask is correct in term of number of selected hypothesis.
                # if rejection_method == 'uniform_negative' :
                    # assert selected_confidence_mask.sum() == batch*number_unconfident+torch.sum(gt_conf_stacked_t==1), "The number of selected hypothesis is not correct."
   
                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
            
            else :
                confidence_loss = torch.tensor(0.)

            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
            #idx_selected

        elif mode == 'awta':

            boltzmann_dist = torch.exp(-dist_matrix/self.temperature) #Shape [batch,Max_sources,num_hyps]

            # normalize the dist
            # Assuming boltzmann_dist is defined and calculated as above
            sums = torch.sum(boltzmann_dist, dim=2, keepdim=True)  # shape [batch,Max_sources,1], sum along the last dimension, keeping dimension
            sums_expanded = sums.expand_as(boltzmann_dist)  # shape [batch,Max_sources,num_hyps], expand the sums to the same shape as the original tensor

            # Create a mask where sums are non-zero
            non_zero_mask = sums != 0
            non_zero_mask_expanded = sums_expanded != 0
            
            # normalize the dist
            boltzmann_dist[non_zero_mask_expanded] = boltzmann_dist[non_zero_mask_expanded]/sums_expanded[non_zero_mask_expanded] #Shape [batch,Max_sources,num_hyps]

            # Reshape the distribution to 2D for multinomial sampling
            boltzmann_dist_flat = boltzmann_dist.view(-1, num_hyps)  # Shape [batch * Max_sources, num_hyps]

            # discard the zero sums indexes for sampling
            # Filter flat distribution to only include rows with non-zero sums
            valid_boltzmann_dist_flat = boltzmann_dist_flat[non_zero_mask.view(-1), :]

            # Sample using multinomial from the valid distributions
            idx_selected_flat = torch.multinomial(valid_boltzmann_dist_flat, 1)  # Shape [num_valid, 1]

            # Initialize the full index tensor with placeholders (-1 or similar)
            
            idx_selected = torch.full((batch, Max_sources, 1), 0, dtype=torch.int64, device=idx_selected_flat.device)  # Shape [batch, Max_sources, 1]
            
            # a = idx_selected[non_zero_mask]
            # # Expand idx_selected_flat to match the original dimensions
            # b = idx_selected_flat.expand_as(idx_selected[non_zero_mask].unsqueeze(-1))
            # idx_selected[non_zero_mask] = b.squeeze(-1)

            # Find indices of valid entries to map back
            device = idx_selected_flat.device
            valid_indices = non_zero_mask[:,:,0]
            valid_indices = non_zero_mask.nonzero(as_tuple=False).to(device)  # Shape [num_valid, 2]
            # Use advanced indexing to assign values
            idx_selected[valid_indices[:, 0], valid_indices[:, 1], 0] = idx_selected_flat.squeeze()

            # ABOVE lines equivalent to 
            # valid_indices = non_zero_mask.nonzero(as_tuple=True)
            # for idx, valid_idx in enumerate(zip(*valid_indices)):
            #     i, j = valid_idx
            #     idx_selected[i, j, 0] = idx_selected_flat[idx]
            #############

            # Sample using multinomial
            awta_dist_matrix = torch.gather(dist_matrix, 2, idx_selected) #Shape [batch,Max_sources,1]
            awta_dist_matrix = torch.squeeze(awta_dist_matrix, dim=2) #Shape [batch,Max_sources]

            mask = (source_activity_target == 1).squeeze(-1) #We create a mask of shape [batch,Max_sources] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            awta_dist_matrix = awta_dist_matrix*mask #[batch,Max_sources], we select only the active sources.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(awta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)
   
        elif mode == 'stable_awta':

            # We select the best hypothesis for each source
            _, idx_selected = torch.min(dist_matrix, dim=2) #wta_dist_matrix of shape [batch,Max_sources]

            boltzmann_dist = torch.exp(-dist_matrix/self.temperature) #Shape [batch,Max_sources,num_hyps]
            boltzmann_dist = boltzmann_dist.detach() # Backpropagation is not performed through the Boltzmann distribution
            sums = torch.sum(boltzmann_dist, dim=2, keepdim=True)  # shape [batch,Max_sources,1], sum along the last dimension, keeping dimension
            sums_expanded = sums.expand_as(boltzmann_dist)  # shape [batch,Max_sources,num_hyps], expand the sums to the same shape as the original tensor

            # Create a mask where sums are non-zero
            non_zero_mask = sums != 0
            non_zero_mask_expanded = sums_expanded != 0
            
            # normalize the dist
            boltzmann_dist[non_zero_mask_expanded] = boltzmann_dist[non_zero_mask_expanded]/sums_expanded[non_zero_mask_expanded] #Shape [batch,Max_sources,num_hyps]

            awta_dist_matrix = boltzmann_dist*dist_matrix #Shape [batch,Max_sources,num_hyps]
            awta_dist_matrix = torch.sum(awta_dist_matrix, dim=-1) #Shape [batch,Max_sources]
            mask = (source_activity_target == 1).squeeze(-1) #We create a mask of shape [batch,Max_sources] for only selecting the actives sources, i.e. those which were not filled with fake values. 
            awta_dist_matrix = awta_dist_matrix*mask #[batch,Max_sources], we select only the active sources.

            # Create tensors to index batch and Max_sources dimensions
            batch_indices = torch.arange(batch, device=conf_pred_stacked.device)[:, None].expand(-1, Max_sources) # Shape (batch, Max_sources)

            # We set the confidences of the selected hypotheses.
            gt_conf_stacked_t[batch_indices[mask], idx_selected[mask]] = 1 #Shape (batch,num_hyps)

            count_non_zeros = torch.sum(mask!=0) #We count the number of actives sources for the computation of the mean (below). 
            
            if count_non_zeros>0 : 
                loss = torch.sum(awta_dist_matrix)/count_non_zeros #We compute the mean of the diff.
                
                selected_confidence_mask = gt_conf_stacked_t == 1 # (batch,num_hyps), this mask will refer to the ground truth of the confidence scores which
                # will be selected for the scoring loss computation. At this point, only the positive hypothesis are selected.   
                unselected_mask = ~selected_confidence_mask # (batch,num_hyps), mask for unselected hypotheses ; this mask will refer to the ground truth of the confidence scores which
                # not are not selected at this point for the scoring loss computation.  

                if rejection_method=='uniform_negative' : 
                    # Generate random indices for unconfident hypotheses, ensuring they are not already selected
                    unconfident_indices = torch.stack([torch.multinomial(unselected_mask[b_idx].float(), number_unconfident, replacement=False) for b_idx in range(batch)]) #(batch,number_unconfident)

                    # Update the confidence mask and ground truth for unconfident hypotheses
                    batch_indices = torch.arange(batch)[:, None].expand(-1, number_unconfident) #(batch,number_unconfident)
                    selected_confidence_mask[batch_indices, unconfident_indices] = True
                    gt_conf_stacked_t[batch_indices, unconfident_indices] = 0 #(Useless) Line added for the sake of completness. 

                elif rejection_method=='all' :

                    selected_confidence_mask = torch.ones_like(selected_confidence_mask).bool() # (batch,num_hyps)

                # assert conf_pred_stacked.all()>0, "The original tensor was affected by the modification" # To check that the original tensor was not affected by the modification. 
                
                # Compute loss only for the selected elements
                confidence_loss = torch.nn.functional.binary_cross_entropy(conf_pred_stacked[selected_confidence_mask], gt_conf_stacked_t[selected_confidence_mask])
               
            else :
                loss = torch.tensor(0.) 
                confidence_loss = torch.tensor(0.)   
            
            sum_losses = torch.add(sum_losses, loss)  
            sum_losses = torch.add(sum_losses, conf_weight*confidence_loss)

        return sum_losses
   