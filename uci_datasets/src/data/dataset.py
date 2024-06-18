import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import sklearn.datasets as datasets

class ToyDataset(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples):
        super(ToyDataset, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        # Sample a point uniformly from the selected section
        if section == 1:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(-1, 0)
        elif section == 2:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(0, 1)
        elif section == 3:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-1, 0)
        else:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        s1 = (-1, 0, -1, 0)
        s2 = (-1, 0, 0, 1)
        s3 = (0, 1, -1, 0)
        s4 = (0, 1, 0, 1)
        
        # Define the probabilities of selecting each section
        p1 = p4 = (1 - t) / 2
        p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                
            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSourcesToyDataset(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples, Max_sources=2,grid_t=False,t=None) :
        super(MultiSourcesToyDataset, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper
            section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

            # Sample a point uniformly from the selected section
            if section == 1:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(-1, 0)
            elif section == 2:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(0, 1)
            elif section == 3:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(-1, 0)
            else:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(0, 1)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(-1, 0)
                elif section == 2:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(0, 1)
                elif section == 3:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(-1, 0)
                else:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(0, 1)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
    
class MultiSourcesToyDataset_modified(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples, Max_sources=2,grid_t=False,t=None,train_mode=True,val_mode=False) :
        super(MultiSourcesToyDataset_modified, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        self.train_mode = train_mode
        self.val_mode = val_mode
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            if self.train_mode is True : 
                t = np.random.uniform(0, 0.7)
            elif self.train_mode is False and self.val_mode is True :      
                t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper
            section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

            # Sample a point uniformly from the selected section
            if section == 1:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(-1, 0)
            elif section == 2:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(0, 1)
            elif section == 3:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(-1, 0)
            else:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(0, 1)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(-1, 0)
                elif section == 2:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(0, 1)
                elif section == 3:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(-1, 0)
                else:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(0, 1)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
   
class InterpolationPToyDataset(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples):
        super(InterpolationPToyDataset, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        # Sample a point uniformly from the selected section
        if section == 1:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(-1, 0)
        elif section == 2:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(0, 1)
        elif section == 3:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-1, 0)
        else:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Define the probabilities of selecting each section
        p1 = p4 = (1 - t) / 2
        p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            elif choice_distribution == 2:
                origin = np.array([0,0])
                x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSourcesPInterpolationToyDataset(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples, Max_sources=2,grid_t=False,t=None) :
        super(MultiSourcesPInterpolationToyDataset, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:
                output[source,0] = np.random.uniform(-1, 1)
                output[source,1] = np.random.uniform(-1, 1)
            elif choice_distribution == 2:
                origin = np.array([0,0])
                output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # # Sample a point uniformly from the selected section
                # if section == 1:
                #     output[source,0] = np.random.uniform(-1, 0)
                #     output[source,1] = np.random.uniform(-1, 0)
                # elif section == 2:
                #     output[source,0] = np.random.uniform(-1, 0)
                #     output[source,1] = np.random.uniform(0, 1)
                # elif section == 3:
                #     output[source,0] = np.random.uniform(0, 1)
                #     output[source,1] = np.random.uniform(-1, 0)
                # else:
                #     output[source,0] = np.random.uniform(0, 1)
                #     output[source,1] = np.random.uniform(0, 1)

                choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                if choice_distribution == 1:
                    output[source,0] = np.random.uniform(-1, 1)
                    output[source,1] = np.random.uniform(-1, 1)
                elif choice_distribution == 2:
                    origin = np.array([0,0])
                    output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
 
class BInterpolationToyDataset(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples):
        super(BInterpolationToyDataset, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.1   
        self.sigma2=0.1
        self.sigma3=0.05
        self.sigma4=0.1

        self.s1 = (-3/4, -1/4, -3/4, -1/4)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1) 
        # s1 = (-1, 0, -1, 0)
        # s1 = (-3/4, -1/4, -3/4, -1/4)
        # s2 = (-1, 0, 0, 1)
        # s3 = (0, 1, -1, 0)
        # s4 = (0, 1, 0, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4

        # Select a section according to the probabilities defined in the paper
        # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

        if choice_distribution == 1:
            # x = np.random.uniform(-1, 1)
            # y = np.random.uniform(-1, 1)
            #################
            # p1 = p4 = (1 - t) / 2
            # p2 = p3 = t / 2

            p1 = p4 = 1/2
            p2 = p3 = 0

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            #################

        elif choice_distribution == 2:

            p1 = p4 = 0
            p2 = p3 = 1/2

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # origin = np.array([0,0]) 
            mean1 = np.array([-1/2,-1/2])
            mean2 = np.array([-1/2,1/2])
            mean3 = np.array([1/2,-1/2])
            mean4 = np.array([1/2,1/2])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # Sample a point uniformly from the selected section
            if section == 1:
                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
            elif section == 2:
                x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
            elif section == 3:
                x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
            elif section == 4:
                x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4
        
        # Define the probabilities of selecting each section
        # p1 = p4 = (1 - t) / 2
        # p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
        
            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            
            # # Sample a point uniformly from the selected section
            # if section == 1:
            #     x = np.random.uniform(s1[0], s1[1])
            #     y = np.random.uniform(s1[2], s1[3])
            # elif section == 2:
            #     x = np.random.uniform(s2[0], s2[1])
            #     y = np.random.uniform(s2[2], s2[3])
            # elif section == 3:
            #     x = np.random.uniform(s3[0], s3[1])
            #     y = np.random.uniform(s3[2], s3[3])
            # elif section == 4:
            #     x = np.random.uniform(s4[0], s4[1])
            #     y = np.random.uniform(s4[2], s4[3])
            # else:
            #     x = np.random.uniform(-1, 1)
            #     y = np.random.uniform(-1, 1)
                
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:
                # x = np.random.uniform(-1, 1)
                # y = np.random.uniform(-1, 1)
                #################
                # p1 = p4 = (1 - t) / 2
                # p2 = p3 = t / 2

                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:

                p1 = p4 = 0
                p2 = p3 = 1/2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # origin = np.array([0,0]) 
                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)
                # x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class BMultiSourcesInterpolationToyDataset(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples, Max_sources=2,grid_t=False,t=None) :
        super(BMultiSourcesInterpolationToyDataset, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.1   
        self.sigma2=0.1
        self.sigma3=0.05
        self.sigma4=0.1

        self.s1 = (-3/4, -1/4, -3/4, -1/4)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1)

        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            # p1 = p4 = (1 - t) / 2
            # p2 = p3 = t / 2
            # section = np.random.choice([1, 2, 3, 4], p=[p1,p2,p3,p4])

            # # Sample a point uniformly from the selected section
            # if section == 1:
            #     output[source,0] = np.random.uniform(-1, 0)
            #     output[source,1] = np.random.uniform(-1, 0)
            # elif section == 2:
            #     output[source,0] = np.random.uniform(-1, 0)
            #     output[source,1] = np.random.uniform(0, 1)
            # elif section == 3:
            #     output[source,0] = np.random.uniform(0, 1)
            #     output[source,1] = np.random.uniform(-1, 0)
            # else:
            #     output[source,0] = np.random.uniform(0, 1)
   
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            # s1 = (-1, 0, -1, 0)
            s1 = self.s1
            s2 = self.s2
            s3 = self.s3
            s4 = self.s4

            if choice_distribution == 1:
                # x = np.random.uniform(-1, 1)
                # y = np.random.uniform(-1, 1)
                #################
                # p1 = p4 = (1 - t) / 2
                # p2 = p3 = t / 2
                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:
                
                p1 = p4 = 0 
                p2 = p3 = 1 / 2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # origin = np.array([0,0]) 
                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

            output[source,0], output[source,1] = x,y

            # if choice_distribution == 1:
            #     output[source,0] = np.random.uniform(-1, 1)
            #     output[source,1] = np.random.uniform(-1, 1)
            # elif choice_distribution == 2:
            #     origin = np.array([0,0])
            #     output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # # Sample a point uniformly from the selected section
                # if section == 1:
                #     output[source,0] = np.random.uniform(-1, 0)
                #     output[source,1] = np.random.uniform(-1, 0)
                # elif section == 2:
                #     output[source,0] = np.random.uniform(-1, 0)
                #     output[source,1] = np.random.uniform(0, 1)
                # elif section == 3:
                #     output[source,0] = np.random.uniform(0, 1)
                #     output[source,1] = np.random.uniform(-1, 0)
                # else:
                #     output[source,0] = np.random.uniform(0, 1)
                #     output[source,1] = np.random.uniform(0, 1)

                # choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                # if choice_distribution == 1:
                #     output[source,0] = np.random.uniform(-1, 1)
                #     output[source,1] = np.random.uniform(-1, 1)
                # elif choice_distribution == 2:
                #     origin = np.array([0,0])
                #     output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

                ################
                choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                # s1 = (-1, 0, -1, 0)
                s1 = self.s1
                s2 = self.s2
                s3 = self.s3
                s4 = self.s4

                if choice_distribution == 1:
                    # x = np.random.uniform(-1, 1)
                    # y = np.random.uniform(-1, 1)
                    #################
                    p1 = p4 = 1/2
                    p2 = p3 = 0

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0] = np.random.uniform(s1[0], s1[1])
                        output[source,1] = np.random.uniform(s1[2], s1[3])
                    elif section == 2:
                        output[source,0] = np.random.uniform(s2[0], s2[1])
                        output[source,1] = np.random.uniform(s2[2], s2[3])
                    elif section == 3:
                        output[source,0] = np.random.uniform(s3[0], s3[1])
                        output[source,1] = np.random.uniform(s3[2], s3[3])
                    elif section == 4:
                        output[source,0] = np.random.uniform(s4[0], s4[1])
                        output[source,1] = np.random.uniform(s4[2], s4[3])
                    else:
                        output[source,0] = np.random.uniform(-1, 1)
                        output[source,1] = np.random.uniform(-1, 1)
                    #################

                elif choice_distribution == 2:

                    p1 = p4 = 0
                    p2 = p3 = 1/2

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # origin = np.array([0,0]) 
                    mean1 = np.array([-1/2,-1/2])
                    mean2 = np.array([-1/2,1/2])
                    mean3 = np.array([1/2,-1/2])
                    mean4 = np.array([1/2,1/2])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                    elif section == 2:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                    elif section == 3:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                    elif section == 4:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

                ################

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

### Single Fixed Gaussian

class single_gauss(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(single_gauss, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.05**(1/2)   

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        mean1 = np.array([0,0])
        # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

        # Sample a point uniformly from the selected section
        x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            mean1 = np.array([0,0])

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_gauss(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_single_gauss, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.05**(1/2)  

        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            mean1 = np.array([0,0])

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            output[source,0], output[source,1] = x,y

            # if choice_distribution == 1:
            #     output[source,0] = np.random.uniform(-1, 1)
            #     output[source,1] = np.random.uniform(-1, 1)
            # elif choice_distribution == 2:
            #     origin = np.array([0,0])
            #     output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                mean1 = np.array([0,0])

                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Fixed Fixed Uniform Law

class single_uniform(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(single_uniform, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.5   

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            # mean1 = np.array([0,0])

            # x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
            x = np.random.uniform(-1,1)
            y = np.random.uniform(-1,1)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_uniform(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_single_uniform, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.5   

        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples

    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            x = np.random.uniform(-1,1)
            y = np.random.uniform(-1,1)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                mean1 = np.array([0,0])

                # x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                x = np.random.uniform(-1,1)
                y = np.random.uniform(-1,1)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
    
# Single Uniform dist to single gaussian 
    
class uniform_to_single_gauss(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(uniform_to_single_gauss, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        # Sample a point uniformly from the selected section
        if section == 1:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(-1, 0)
        elif section == 2:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(0, 1)
        elif section == 3:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-1, 0)
        else:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Define the probabilities of selecting each section
        p1 = p4 = (1 - t) / 2
        p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            elif choice_distribution == 2:
                origin = np.array([0,0])
                x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_uniform_to_single_gauss(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_uniform_to_single_gauss, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:
                output[source,0] = np.random.uniform(-1, 1)
                output[source,1] = np.random.uniform(-1, 1)
            elif choice_distribution == 2:
                origin = np.array([0,0])
                output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper

                choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                if choice_distribution == 1:
                    output[source,0] = np.random.uniform(-1, 1)
                    output[source,1] = np.random.uniform(-1, 1)
                elif choice_distribution == 2:
                    origin = np.array([0,0])
                    output[source,0], output[source,1] = np.random.multivariate_normal(origin, np.eye(2)*0.05)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
   
# Mixture of Uniform to mixture of gaussians
   
class mixture_uni_to_gaussians(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(mixture_uni_to_gaussians, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.1   
        self.sigma2=0.1
        self.sigma3=0.1
        self.sigma4=0.1

        self.s1 = (-1, 0, -1,0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1) 

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4

        # Select a section according to the probabilities defined in the paper
        # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

        if choice_distribution == 1:

            p1 = p4 = 1/2
            p2 = p3 = 0

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            #################

        elif choice_distribution == 2:

            p1 = p4 = 0
            p2 = p3 = 1/2

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # origin = np.array([0,0]) 
            mean1 = np.array([-1/2,-1/2])
            mean2 = np.array([-1/2,1/2])
            mean3 = np.array([1/2,-1/2])
            mean4 = np.array([1/2,1/2])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # Sample a point uniformly from the selected section
            if section == 1:
                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
            elif section == 2:
                x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
            elif section == 3:
                x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
            elif section == 4:
                x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4
        
        # Define the probabilities of selecting each section
        # p1 = p4 = (1 - t) / 2
        # p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
       
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:

                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:

                p1 = p4 = 0
                p2 = p3 = 1/2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # origin = np.array([0,0]) 
                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)
                # x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_mixture_uni_to_gaussians(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_mixture_uni_to_gaussians, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.1   
        self.sigma2=0.1
        self.sigma3=0.1
        self.sigma4=0.1

        self.s1 = (-1, 0, -1, 0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1)

        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True   
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            s1 = self.s1
            s2 = self.s2
            s3 = self.s3
            s4 = self.s4

            if choice_distribution == 1:

                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:
                
                p1 = p4 = 0 
                p2 = p3 = 1 / 2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):

        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True

                ################
                choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                s1 = self.s1
                s2 = self.s2
                s3 = self.s3
                s4 = self.s4

                if choice_distribution == 1:
                    # x = np.random.uniform(-1, 1)
                    # y = np.random.uniform(-1, 1)
                    #################
                    p1 = p4 = 1/2
                    p2 = p3 = 0

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0] = np.random.uniform(s1[0], s1[1])
                        output[source,1] = np.random.uniform(s1[2], s1[3])
                    elif section == 2:
                        output[source,0] = np.random.uniform(s2[0], s2[1])
                        output[source,1] = np.random.uniform(s2[2], s2[3])
                    elif section == 3:
                        output[source,0] = np.random.uniform(s3[0], s3[1])
                        output[source,1] = np.random.uniform(s3[2], s3[3])
                    elif section == 4:
                        output[source,0] = np.random.uniform(s4[0], s4[1])
                        output[source,1] = np.random.uniform(s4[2], s4[3])
                    else:
                        output[source,0] = np.random.uniform(-1, 1)
                        output[source,1] = np.random.uniform(-1, 1)
                    #################

                elif choice_distribution == 2:

                    p1 = p4 = 0
                    p2 = p3 = 1/2

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # origin = np.array([0,0]) 
                    mean1 = np.array([-1/2,-1/2])
                    mean2 = np.array([-1/2,1/2])
                    mean3 = np.array([1/2,-1/2])
                    mean4 = np.array([1/2,1/2])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                    elif section == 2:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                    elif section == 3:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                    elif section == 4:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

                ################

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
    
# Mixture uni to gaussians

# Mixture of Uniform to mixture of gaussians
   
class mixture_uni_to_gaussians_v2(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(mixture_uni_to_gaussians_v2, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.1   
        self.sigma2=0.25
        self.sigma3=0.05
        self.sigma4=0.1

        self.s1 = (-1, 0, -1,0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1) 

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4

        # Select a section according to the probabilities defined in the paper
        # section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

        if choice_distribution == 1:

            p1 = p4 = 1/2
            p2 = p3 = 0

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            #################

        elif choice_distribution == 2:

            p1 = p4 = 0
            p2 = p3 = 1/2

            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # origin = np.array([0,0]) 
            mean1 = np.array([-1/2,-1/2])
            mean2 = np.array([-1/2,1/2])
            mean3 = np.array([1/2,-1/2])
            mean4 = np.array([1/2,1/2])

            # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

            # Sample a point uniformly from the selected section
            if section == 1:
                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
            elif section == 2:
                x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
            elif section == 3:
                x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
            elif section == 4:
                x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        # s1 = (-1, 0, -1, 0)
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        s4 = self.s4
        
        # Define the probabilities of selecting each section
        # p1 = p4 = (1 - t) / 2
        # p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
       
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            if choice_distribution == 1:

                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:

                p1 = p4 = 0
                p2 = p3 = 1/2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                # origin = np.array([0,0]) 
                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)
                # x, y = np.random.multivariate_normal(origin, np.eye(2)*0.05)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_mixture_uni_to_gaussians_v2(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_mixture_uni_to_gaussians_v2, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.1   
        self.sigma2=0.25
        self.sigma3=0.05
        self.sigma4=0.1

        self.s1 = (-1, 0, -1, 0)
        self.s2 = (-1, 0, 0, 1)
        self.s3 = (0, 1, -1, 0)
        self.s4 = (0, 1, 0, 1)

        self.grid_t = grid_t
        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True   
            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            s1 = self.s1
            s2 = self.s2
            s3 = self.s3
            s4 = self.s4

            if choice_distribution == 1:

                p1 = p4 = 1/2
                p2 = p3 = 0

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
                # Sample a point uniformly from the selected section
                if section == 1:
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section == 2:
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
                elif section == 3:
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section == 4:
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
                else:
                    x = np.random.uniform(-1, 1)
                    y = np.random.uniform(-1, 1)
                #################

            elif choice_distribution == 2:
                
                p1 = p4 = 0 
                p2 = p3 = 1 / 2

                section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                mean1 = np.array([-1/2,-1/2])
                mean2 = np.array([-1/2,1/2])
                mean3 = np.array([1/2,-1/2])
                mean4 = np.array([1/2,1/2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                elif section == 2:
                    x,y = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                elif section == 3:
                    x,y = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                elif section == 4:
                    x,y = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):

        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True

                ################
                choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

                s1 = self.s1
                s2 = self.s2
                s3 = self.s3
                s4 = self.s4

                if choice_distribution == 1:
                    # x = np.random.uniform(-1, 1)
                    # y = np.random.uniform(-1, 1)
                    #################
                    p1 = p4 = 1/2
                    p2 = p3 = 0

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0] = np.random.uniform(s1[0], s1[1])
                        output[source,1] = np.random.uniform(s1[2], s1[3])
                    elif section == 2:
                        output[source,0] = np.random.uniform(s2[0], s2[1])
                        output[source,1] = np.random.uniform(s2[2], s2[3])
                    elif section == 3:
                        output[source,0] = np.random.uniform(s3[0], s3[1])
                        output[source,1] = np.random.uniform(s3[2], s3[3])
                    elif section == 4:
                        output[source,0] = np.random.uniform(s4[0], s4[1])
                        output[source,1] = np.random.uniform(s4[2], s4[3])
                    else:
                        output[source,0] = np.random.uniform(-1, 1)
                        output[source,1] = np.random.uniform(-1, 1)
                    #################

                elif choice_distribution == 2:

                    p1 = p4 = 0
                    p2 = p3 = 1/2

                    section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

                    # origin = np.array([0,0]) 
                    mean1 = np.array([-1/2,-1/2])
                    mean2 = np.array([-1/2,1/2])
                    mean3 = np.array([1/2,-1/2])
                    mean4 = np.array([1/2,1/2])

                    # Sample a point uniformly from the selected section
                    if section == 1:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)
                    elif section == 2:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean2, np.eye(2)*self.sigma2**2)
                    elif section == 3:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean3, np.eye(2)*self.sigma3**2)
                    elif section == 4:
                        output[source,0],output[source,1] = np.random.multivariate_normal(mean4, np.eye(2)*self.sigma4**2)

                ################

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Mixture of uniform to uniform 
    
class mixture_uni_to_uni(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(mixture_uni_to_uni, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

        # Sample a point uniformly from the selected section
        if section == 1:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(-1, 0)
        elif section == 2:
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(0, 1)
        elif section == 3:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-1, 0)
        else:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        s1 = (-1, 0, -1, 0)
        s2 = (-1, 0, 0, 1)
        s3 = (0, 1, -1, 0)
        s4 = (0, 1, 0, 1)
        
        # Define the probabilities of selecting each section
        p1 = p4 = (1 - t) / 2
        p2 = p3 = t / 2
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Select a section
            section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])
            
            # Sample a point uniformly from the selected section
            if section == 1:
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section == 2:
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
            elif section == 3:
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section == 4:
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
            else:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                
            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_mixture_uni_to_uni(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_mixture_uni_to_uni, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper
            section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

            # Sample a point uniformly from the selected section
            if section == 1:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(-1, 0)
            elif section == 2:
                output[source,0] = np.random.uniform(-1, 0)
                output[source,1] = np.random.uniform(0, 1)
            elif section == 3:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(-1, 0)
            else:
                output[source,0] = np.random.uniform(0, 1)
                output[source,1] = np.random.uniform(0, 1)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                section = np.random.choice([1, 2, 3, 4], p=[(1 - t) / 2, t / 2, t / 2, (1 - t) / 2])

                # Sample a point uniformly from the selected section
                if section == 1:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(-1, 0)
                elif section == 2:
                    output[source,0] = np.random.uniform(-1, 0)
                    output[source,1] = np.random.uniform(0, 1)
                elif section == 3:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(-1, 0)
                else:
                    output[source,0] = np.random.uniform(0, 1)
                    output[source,1] = np.random.uniform(0, 1)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Two moons
    
class two_moons(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=10000):
        super(two_moons, self).__init__()
        self.n_samples = n_samples
        points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1) 
        self.points = points

        self.points[:,0] = 2*(points[:,0]-(points[:,0].min()))/(points[:,0].max()-points[:,0].min())-1
        self.points[:,1] = 2*(points[:,1]-points[:,1].min())/(points[:,1].max()-points[:,1].min())-1

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        index = np.random.randint(0, self.n_samples)

        x = self.points[index,0]
        y = self.points[index,1]

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Generate n_samples samples

        samples, _ = datasets.make_moons(n_samples=n_samples, noise=0.1) 
        samples[:,0] = 2*(samples[:,0]-(samples[:,0].min()))/(samples[:,0].max()-samples[:,0].min())-1
        samples[:,1] = 2*(samples[:,1]-samples[:,1].min())/(samples[:,1].max()-samples[:,1].min())-1

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_two_moons(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=10000, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_two_moons, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1) 
        self.points = points

        self.points[:,0] = 2*(points[:,0]-(points[:,0].min()))/(points[:,0].max()-points[:,0].min())-1
        self.points[:,1] = 2*(points[:,1]-points[:,1].min())/(points[:,1].max()-points[:,1].min())-1

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            index = np.random.randint(0, self.n_samples)

            output[source,0] = self.points[index,0]
            output[source,1] = self.points[index,1]

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        points, _ = datasets.make_moons(n_samples=n_samples, noise=0.1) 
        points = points

        points[:,0] = 2*(points[:,0]-(points[:,0].min()))/(points[:,0].max()-points[:,0].min())-1
        points[:,1] = 2*(points[:,1]-points[:,1].min())/(points[:,1].max()-points[:,1].min())-1

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
        
                index = np.random.randint(0, n_samples)

                samples[i,source,0] = points[index,0]
                samples[i,source,1] = points[index,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# TODO Rotating Two moons
 
class rotating_two_moons(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=10000):
        super(rotating_two_moons, self).__init__()
        self.n_samples = n_samples
        initial_points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1) 
        self.initial_points = initial_points

        self.initial_points[:,0] = 2*(initial_points[:,0]-(initial_points[:,0].min()))/(initial_points[:,0].max()-initial_points[:,0].min())-1
        self.initial_points[:,1] = 2*(initial_points[:,1]-initial_points[:,1].min())/(initial_points[:,1].max()-initial_points[:,1].min())-1

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        angle = t*2*np.pi

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        index = np.random.randint(0, self.n_samples)

        vector_sampled = self.initial_points[index,:]
        rotated_sample = np.dot(vector_sampled,rotation_matrix)

        x = rotated_sample[0]
        y = rotated_sample[1]

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Generate n_samples samples

        samples, _ = datasets.make_moons(n_samples=n_samples, noise=0.1) 
        samples[:,0] = 2*(samples[:,0]-(samples[:,0].min()))/(samples[:,0].max()-samples[:,0].min())-1
        samples[:,1] = 2*(samples[:,1]-samples[:,1].min())/(samples[:,1].max()-samples[:,1].min())-1
       
        angle = t*2*np.pi
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        rotated_samples = np.dot(samples,rotation_matrix)

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return rotated_samples

class MultiSources_rotating_two_moons(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=10000, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_rotating_two_moons, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        initial_points, _ = datasets.make_moons(n_samples=self.n_samples, noise=0.1) 
        self.initial_points = initial_points

        self.initial_points[:,0] = 2*(initial_points[:,0]-(initial_points[:,0].min()))/(initial_points[:,0].max()-initial_points[:,0].min())-1
        self.initial_points[:,1] = 2*(initial_points[:,1]-initial_points[:,1].min())/(initial_points[:,1].max()-initial_points[:,1].min())-1

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            angle = t*2*np.pi

            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            index = np.random.randint(0, self.n_samples)

            vector_sampled = self.initial_points[index,:]
            rotated_sample = np.dot(vector_sampled,rotation_matrix)

            output[source,0] = rotated_sample[0]
            output[source,1] = rotated_sample[1]

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        initial_points, _ = datasets.make_moons(n_samples=n_samples, noise=0.1) 
        initial_points = initial_points

        initial_points[:,0] = 2*(initial_points[:,0]-(initial_points[:,0].min()))/(initial_points[:,0].max()-initial_points[:,0].min())-1
        initial_points[:,1] = 2*(initial_points[:,1]-initial_points[:,1].min())/(initial_points[:,1].max()-initial_points[:,1].min())-1

        for i in range(n_samples):

            N_sources = 1

            angle = t*2*np.pi

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                ###################

                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

                index = np.random.randint(0, self.n_samples)

                vector_sampled = initial_points[index,:]
                rotated_sample = np.dot(vector_sampled,rotation_matrix)

                samples[i,source,0] = rotated_sample[0]
                samples[i,source,1] = rotated_sample[1]
                ###################

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Fixed damier  
  
class fixed_damier(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(fixed_damier, self).__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
        section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

        s1 = (-0.5, 0, 0.5, 1)
        s2 = (0.5, 1, 0.5, 1)
        s3 = (-1,-0.5,0,0.5)
        s4 = (0,0.5,0,0.5)
        s5 = (-0.5,0,-0.5,0)
        s6 = (0.5,1,-0.5,0)
        s7 = (-1,-0.5,-1,-0.5)
        s8 = (0,0.5,-1,-0.5)

        if section_rows == 1 :
            if section_cols == 1 :
                x = np.random.uniform(s1[0], s1[1])
                y = np.random.uniform(s1[2], s1[3])
            elif section_cols == 2 :
                x = np.random.uniform(s2[0], s2[1])
                y = np.random.uniform(s2[2], s2[3])
        elif section_rows == 2 :
            if section_cols == 1 :
                x = np.random.uniform(s3[0], s3[1])
                y = np.random.uniform(s3[2], s3[3])
            elif section_cols == 2 :
                x = np.random.uniform(s4[0], s4[1])
                y = np.random.uniform(s4[2], s4[3])
        elif section_rows == 3 :
            if section_cols == 1 :
                x = np.random.uniform(s5[0], s5[1])
                y = np.random.uniform(s5[2], s5[3])
            elif section_cols == 2 :
                x = np.random.uniform(s6[0], s6[1])
                y = np.random.uniform(s6[2], s6[3])
        elif section_rows == 4 :
            if section_cols == 1 :
                x = np.random.uniform(s7[0], s7[1])
                y = np.random.uniform(s7[2], s7[3])
            elif section_cols == 2 :
                x = np.random.uniform(s8[0], s8[1])
                y = np.random.uniform(s8[2], s8[3])
    
        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        s1 = (-0.5, 0, 0.5, 1)
        s2 = (0.5, 1, 0.5, 1)
        s3 = (-1,-0.5,0,0.5)
        s4 = (0,0.5,0,0.5)
        s5 = (-0.5,0,-0.5,0)
        s6 = (0.5,1,-0.5,0)
        s7 = (-1,-0.5,-1,-0.5)
        s8 = (0,0.5,-1,-0.5)

        samples = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Select a section
            # Select a section according to the probabilities defined in the paper
            section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

            s1 = (-0.5, 0, 0.5, 1)
            s2 = (0.5, 1, 0.5, 1)
            s3 = (-1,-0.5,0,0.5)
            s4 = (0,0.5,0,0.5)
            s5 = (-0.5,0,-0.5,0)
            s6 = (0.5,1,-0.5,0)
            s7 = (-1,-0.5,-1,-0.5)
            s8 = (0,0.5,-1,-0.5)

            if section_rows == 1 :
                if section_cols == 1 :
                    x = np.random.uniform(s1[0], s1[1])
                    y = np.random.uniform(s1[2], s1[3])
                elif section_cols == 2 :
                    x = np.random.uniform(s2[0], s2[1])
                    y = np.random.uniform(s2[2], s2[3])
            elif section_rows == 2 :
                if section_cols == 1 :
                    x = np.random.uniform(s3[0], s3[1])
                    y = np.random.uniform(s3[2], s3[3])
                elif section_cols == 2 :
                    x = np.random.uniform(s4[0], s4[1])
                    y = np.random.uniform(s4[2], s4[3])
            elif section_rows == 3 :
                if section_cols == 1 :
                    x = np.random.uniform(s5[0], s5[1])
                    y = np.random.uniform(s5[2], s5[3])
                elif section_cols == 2 :
                    x = np.random.uniform(s6[0], s6[1])
                    y = np.random.uniform(s6[2], s6[3])
            elif section_rows == 4 :
                if section_cols == 1 :
                    x = np.random.uniform(s7[0], s7[1])
                    y = np.random.uniform(s7[2], s7[3])
                elif section_cols == 2 :
                    x = np.random.uniform(s8[0], s8[1])
                    y = np.random.uniform(s8[2], s8[3])
                
            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_fixed_damier(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_fixed_damier, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True

            ##################
            # Select a section according to the probabilities defined in the paper
            section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

            s1 = (-0.5, 0, 0.5, 1)
            s2 = (0.5, 1, 0.5, 1)
            s3 = (-1,-0.5,0,0.5)
            s4 = (0,0.5,0,0.5)
            s5 = (-0.5,0,-0.5,0)
            s6 = (0.5,1,-0.5,0)
            s7 = (-1,-0.5,-1,-0.5)
            s8 = (0,0.5,-1,-0.5)

            if section_rows == 1 :
                if section_cols == 1 :
                    output[source,0] = np.random.uniform(s1[0], s1[1])
                    output[source,1] = np.random.uniform(s1[2], s1[3])
                elif section_cols == 2 :
                    output[source,0] = np.random.uniform(s2[0], s2[1])
                    output[source,1] = np.random.uniform(s2[2], s2[3])
            elif section_rows == 2 :
                if section_cols == 1 :
                    output[source,0] = np.random.uniform(s3[0], s3[1])
                    output[source,1] = np.random.uniform(s3[2], s3[3])
                elif section_cols == 2 :
                    output[source,0] = np.random.uniform(s4[0], s4[1])
                    output[source,1] = np.random.uniform(s4[2], s4[3])
            elif section_rows == 3 :
                if section_cols == 1 :
                    output[source,0] = np.random.uniform(s5[0], s5[1])
                    output[source,1] = np.random.uniform(s5[2], s5[3])
                elif section_cols == 2 :
                    output[source,0] = np.random.uniform(s6[0], s6[1])
                    output[source,1] = np.random.uniform(s6[2], s6[3])
            elif section_rows == 4 :
                if section_cols == 1 :
                    output[source,0] = np.random.uniform(s7[0], s7[1])
                    output[source,1] = np.random.uniform(s7[2], s7[3])
                elif section_cols == 2 :
                    output[source,0] = np.random.uniform(s8[0], s8[1])
                    output[source,1] = np.random.uniform(s8[2], s8[3])

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                ####################
                section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
                section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

                s1 = (-0.5, 0, 0.5, 1)
                s2 = (0.5, 1, 0.5, 1)
                s3 = (-1,-0.5,0,0.5)
                s4 = (0,0.5,0,0.5)
                s5 = (-0.5,0,-0.5,0)
                s6 = (0.5,1,-0.5,0)
                s7 = (-1,-0.5,-1,-0.5)
                s8 = (0,0.5,-1,-0.5)

                if section_rows == 1 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(s1[0], s1[1])
                        output[source,1] = np.random.uniform(s1[2], s1[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(s2[0], s2[1])
                        output[source,1] = np.random.uniform(s2[2], s2[3])
                elif section_rows == 2 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(s3[0], s3[1])
                        output[source,1] = np.random.uniform(s3[2], s3[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(s4[0], s4[1])
                        output[source,1] = np.random.uniform(s4[2], s4[3])
                elif section_rows == 3 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(s5[0], s5[1])
                        output[source,1] = np.random.uniform(s5[2], s5[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(s6[0], s6[1])
                        output[source,1] = np.random.uniform(s6[2], s6[3])
                elif section_rows == 4 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(s7[0], s7[1])
                        output[source,1] = np.random.uniform(s7[2], s7[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(s8[0], s8[1])
                        output[source,1] = np.random.uniform(s8[2], s8[3])
                ####################

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Changing damier
  
class changing_damier(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(changing_damier, self).__init__()
        self.n_samples = n_samples

        self.s1 = (-0.5, 0, 0.5, 1)
        self.s2 = (0.5, 1, 0.5, 1)
        self.s3 = (-1,-0.5,0,0.5)
        self.s4 = (0,0.5,0,0.5)
        self.s5 = (-0.5,0,-0.5,0)
        self.s6 = (0.5,1,-0.5,0)
        self.s7 = (-1,-0.5,-1,-0.5)
        self.s8 = (0,0.5,-1,-0.5)

        self.c1 = (-1,-0.5,0.5,1)
        self.c2 = (0,0.5,0.5,1)
        self.c3 = (-0.5,0,0,0.5)
        self.c4 = (0.5,1,0,0.5)
        self.c5 = (-1,-0.5,-0.5,0)
        self.c6 = (0,0.5,-0.5,0)
        self.c7 = (-0.5,0,-1,-0.5)
        self.c8 = (0.5,1,-1,-0.5)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Select a section according to the probabilities defined in the paper
        choice_damier = np.random.choice(range(1, 3), p=[1-t,t])
        
        section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
        section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

        if choice_damier == 1 : 

            if section_rows == 1 :
                if section_cols == 1 :
                    x = np.random.uniform(self.s1[0], self.s1[1])
                    y = np.random.uniform(self.s1[2], self.s1[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.s2[0], self.s2[1])
                    y = np.random.uniform(self.s2[2], self.s2[3])
            elif section_rows == 2 :
                if section_cols == 1 :
                    x = np.random.uniform(self.s3[0], self.s3[1])
                    y = np.random.uniform(self.s3[2], self.s3[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.s4[0], self.s4[1])
                    y = np.random.uniform(self.s4[2], self.s4[3])
            elif section_rows == 3 :
                if section_cols == 1 :
                    x = np.random.uniform(self.s5[0], self.s5[1])
                    y = np.random.uniform(self.s5[2], self.s5[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.s6[0], self.s6[1])
                    y = np.random.uniform(self.s6[2], self.s6[3])
            elif section_rows == 4 :
                if section_cols == 1 :
                    x = np.random.uniform(self.s7[0], self.s7[1])
                    y = np.random.uniform(self.s7[2], self.s7[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.s8[0], self.s8[1])
                    y = np.random.uniform(self.s8[2], self.s8[3])

        elif choice_damier == 2 : 

            if section_rows == 1 :
                if section_cols == 1 :
                    x = np.random.uniform(self.c1[0], self.c1[1])
                    y = np.random.uniform(self.c1[2], self.c1[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.c2[0], self.c2[1])
                    y = np.random.uniform(self.c2[2], self.c2[3])
            elif section_rows == 2 :
                if section_cols == 1 :
                    x = np.random.uniform(self.c3[0], self.c3[1])
                    y = np.random.uniform(self.c3[2], self.c3[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.c4[0], self.c4[1])
                    y = np.random.uniform(self.c4[2], self.c4[3])
            elif section_rows == 3 :
                if section_cols == 1 :
                    x = np.random.uniform(self.c5[0], self.c5[1])
                    y = np.random.uniform(self.c5[2], self.c5[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.c6[0], self.c6[1])
                    y = np.random.uniform(self.c6[2], self.c6[3])
            elif section_rows == 4 :
                if section_cols == 1 :
                    x = np.random.uniform(self.c7[0], self.c7[1])
                    y = np.random.uniform(self.c7[2], self.c7[3])
                elif section_cols == 2 :
                    x = np.random.uniform(self.c8[0], self.c8[1])
                    y = np.random.uniform(self.c8[2], self.c8[3])
    
        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""

        samples = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Select a section
            # Select a section according to the probabilities defined in the paper
           # Select a section according to the probabilities defined in the paper
            choice_damier = np.random.choice(range(1, 3), p=[1-t,t])
            
            section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

            if choice_damier == 1 : 

                if section_rows == 1 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.s1[0], self.s1[1])
                        y = np.random.uniform(self.s1[2], self.s1[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.s2[0], self.s2[1])
                        y = np.random.uniform(self.s2[2], self.s2[3])
                elif section_rows == 2 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.s3[0], self.s3[1])
                        y = np.random.uniform(self.s3[2], self.s3[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.s4[0], self.s4[1])
                        y = np.random.uniform(self.s4[2], self.s4[3])
                elif section_rows == 3 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.s5[0], self.s5[1])
                        y = np.random.uniform(self.s5[2], self.s5[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.s6[0], self.s6[1])
                        y = np.random.uniform(self.s6[2], self.s6[3])
                elif section_rows == 4 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.s7[0], self.s7[1])
                        y = np.random.uniform(self.s7[2], self.s7[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.s8[0], self.s8[1])
                        y = np.random.uniform(self.s8[2], self.s8[3])

            elif choice_damier == 2 : 

                if section_rows == 1 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.c1[0], self.c1[1])
                        y = np.random.uniform(self.c1[2], self.c1[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.c2[0], self.c2[1])
                        y = np.random.uniform(self.c2[2], self.c2[3])
                elif section_rows == 2 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.c3[0], self.c3[1])
                        y = np.random.uniform(self.c3[2], self.c3[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.c4[0], self.c4[1])
                        y = np.random.uniform(self.c4[2], self.c4[3])
                elif section_rows == 3 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.c5[0], self.c5[1])
                        y = np.random.uniform(self.c5[2], self.c5[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.c6[0], self.c6[1])
                        y = np.random.uniform(self.c6[2], self.c6[3])
                elif section_rows == 4 :
                    if section_cols == 1 :
                        x = np.random.uniform(self.c7[0], self.c7[1])
                        y = np.random.uniform(self.c7[2], self.c7[3])
                    elif section_cols == 2 :
                        x = np.random.uniform(self.c8[0], self.c8[1])
                        y = np.random.uniform(self.c8[2], self.c8[3])
                
            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_changing_damier(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_changing_damier, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        self.s1 = (-0.5, 0, 0.5, 1)
        self.s2 = (0.5, 1, 0.5, 1)
        self.s3 = (-1,-0.5,0,0.5)
        self.s4 = (0,0.5,0,0.5)
        self.s5 = (-0.5,0,-0.5,0)
        self.s6 = (0.5,1,-0.5,0)
        self.s7 = (-1,-0.5,-1,-0.5)
        self.s8 = (0,0.5,-1,-0.5)

        self.c1 = (-1,-0.5,0.5,1)
        self.c2 = (0,0.5,0.5,1)
        self.c3 = (-0.5,0,0,0.5)
        self.c4 = (0.5,1,0,0.5)
        self.c5 = (-1,-0.5,-0.5,0)
        self.c6 = (0,0.5,-0.5,0)
        self.c7 = (-0.5,0,-1,-0.5)
        self.c8 = (0.5,1,-1,-0.5)

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True

            ##################
            choice_damier = np.random.choice(range(1, 3), p=[1-t,t])
            
            section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
            section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

            if choice_damier == 1 : 

                if section_rows == 1 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.s1[0], self.s1[1])
                        output[source,1] = np.random.uniform(self.s1[2], self.s1[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.s2[0], self.s2[1])
                        output[source,1] = np.random.uniform(self.s2[2], self.s2[3])
                elif section_rows == 2 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.s3[0], self.s3[1])
                        output[source,1] = np.random.uniform(self.s3[2], self.s3[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.s4[0], self.s4[1])
                        output[source,1] = np.random.uniform(self.s4[2], self.s4[3])
                elif section_rows == 3 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.s5[0], self.s5[1])
                        output[source,1] = np.random.uniform(self.s5[2], self.s5[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.s6[0], self.s6[1])
                        output[source,1] = np.random.uniform(self.s6[2], self.s6[3])
                elif section_rows == 4 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.s7[0], self.s7[1])
                        output[source,1] = np.random.uniform(self.s7[2], self.s7[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.s8[0], self.s8[1])
                        output[source,1] = np.random.uniform(self.s8[2], self.s8[3])

            elif choice_damier == 2 : 

                if section_rows == 1 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.c1[0], self.c1[1])
                        output[source,1] = np.random.uniform(self.c1[2], self.c1[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.c2[0], self.c2[1])
                        output[source,1] = np.random.uniform(self.c2[2], self.c2[3])
                elif section_rows == 2 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.c3[0], self.c3[1])
                        output[source,1] = np.random.uniform(self.c3[2], self.c3[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.c4[0], self.c4[1])
                        output[source,1] = np.random.uniform(self.c4[2], self.c4[3])
                elif section_rows == 3 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.c5[0], self.c5[1])
                        output[source,1] = np.random.uniform(self.c5[2], self.c5[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.c6[0], self.c6[1])
                        output[source,1] = np.random.uniform(self.c6[2], self.c6[3])
                elif section_rows == 4 :
                    if section_cols == 1 :
                        output[source,0] = np.random.uniform(self.c7[0], self.c7[1])
                        output[source,1] = np.random.uniform(self.c7[2], self.c7[3])
                    elif section_cols == 2 :
                        output[source,0] = np.random.uniform(self.c8[0], self.c8[1])
                        output[source,1] = np.random.uniform(self.c8[2], self.c8[3])
            
        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
                ####################
                ##################
                choice_damier = np.random.choice(range(1, 3), p=[1-t,t])
                
                section_rows = np.random.choice(range(1, 5), p=[1/4] * 4)
                section_cols = np.random.choice(range(1, 3), p=[1/2] * 2)

                if choice_damier == 1 : 

                    if section_rows == 1 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.s1[0], self.s1[1])
                            output[source,1] = np.random.uniform(self.s1[2], self.s1[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.s2[0], self.s2[1])
                            output[source,1] = np.random.uniform(self.s2[2], self.s2[3])
                    elif section_rows == 2 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.s3[0], self.s3[1])
                            output[source,1] = np.random.uniform(self.s3[2], self.s3[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.s4[0], self.s4[1])
                            output[source,1] = np.random.uniform(self.s4[2], self.s4[3])
                    elif section_rows == 3 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.s5[0], self.s5[1])
                            output[source,1] = np.random.uniform(self.s5[2], self.s5[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.s6[0], self.s6[1])
                            output[source,1] = np.random.uniform(self.s6[2], self.s6[3])
                    elif section_rows == 4 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.s7[0], self.s7[1])
                            output[source,1] = np.random.uniform(self.s7[2], self.s7[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.s8[0], self.s8[1])
                            output[source,1] = np.random.uniform(self.s8[2], self.s8[3])

                elif choice_damier == 2 : 

                    if section_rows == 1 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.c1[0], self.c1[1])
                            output[source,1] = np.random.uniform(self.c1[2], self.c1[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.c2[0], self.c2[1])
                            output[source,1] = np.random.uniform(self.c2[2], self.c2[3])
                    elif section_rows == 2 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.c3[0], self.c3[1])
                            output[source,1] = np.random.uniform(self.c3[2], self.c3[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.c4[0], self.c4[1])
                            output[source,1] = np.random.uniform(self.c4[2], self.c4[3])
                    elif section_rows == 3 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.c5[0], self.c5[1])
                            output[source,1] = np.random.uniform(self.c5[2], self.c5[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.c6[0], self.c6[1])
                            output[source,1] = np.random.uniform(self.c6[2], self.c6[3])
                    elif section_rows == 4 :
                        if section_cols == 1 :
                            output[source,0] = np.random.uniform(self.c7[0], self.c7[1])
                            output[source,1] = np.random.uniform(self.c7[2], self.c7[3])
                        elif section_cols == 2 :
                            output[source,0] = np.random.uniform(self.c8[0], self.c8[1])
                            output[source,1] = np.random.uniform(self.c8[2], self.c8[3])
                ####################

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
    
# Fixed Swiss roll
         
class swiss_roll(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=10000):
        super(swiss_roll, self).__init__()
        self.n_samples = n_samples
        points, _ = datasets.make_swiss_roll(n_samples=self.n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        self.points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        index = np.random.randint(0, self.n_samples)

        x = self.points[index,0]
        y = self.points[index,1]

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Generate n_samples samples

        samples, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1) 

        samples[:,0] = 1.5*(samples[:,0] - samples[:,0].min()) / (samples[:,0].max() - samples[:,0].min()) - 0.75
        samples[:,1] = 1.5*(samples[:,1] - samples[:,1].min()) / (samples[:,1].max() - samples[:,1].min()) - 0.75
        samples[:,2] = 1.5*(samples[:,2] - samples[:,2].min()) / (samples[:,2].max() - samples[:,2].min()) - 0.75

        samples = np.concatenate((samples[:,0].reshape(-1,1), samples[:,2].reshape(-1,1)), axis=1)

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_swiss_roll(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=10000, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_swiss_roll, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        points, _ = datasets.make_swiss_roll(n_samples=self.n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        self.points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            index = np.random.randint(0, self.n_samples)

            output[source,0] = self.points[index,0]
            output[source,1] = self.points[index,1]

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        points, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

        for i in range(n_samples):

            N_sources = 1

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
        
                index = np.random.randint(0, n_samples)

                samples[i,source,0] = points[index,0]
                samples[i,source,1] = points[index,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
   
# Rotating Swiss roll
      
class rotating_swiss_roll(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=10000):
        super(rotating_swiss_roll, self).__init__()
        self.n_samples = n_samples
        points, _ = datasets.make_swiss_roll(n_samples=self.n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        self.points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        index = np.random.randint(0, self.n_samples)

        x = self.points[index,0]
        y = self.points[index,1]

        theta = t * 2 * np.pi

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x, y = np.dot(rotation_matrix, np.array([x, y]))

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section
        
        # Generate n_samples samples

        samples, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1) 

        samples[:,0] = 1.5*(samples[:,0] - samples[:,0].min()) / (samples[:,0].max() - samples[:,0].min()) - 0.75
        samples[:,1] = 1.5*(samples[:,1] - samples[:,1].min()) / (samples[:,1].max() - samples[:,1].min()) - 0.75
        samples[:,2] = 1.5*(samples[:,2] - samples[:,2].min()) / (samples[:,2].max() - samples[:,2].min()) - 0.75

        samples = np.concatenate((samples[:,0].reshape(-1,1), samples[:,2].reshape(-1,1)), axis=1)

        theta = t * 2 * np.pi

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        samples = np.dot(samples, rotation_matrix)

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

class MultiSources_rotating_swiss_roll(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=10000, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_rotating_swiss_roll, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.grid_t = grid_t
        if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
            self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

        points, _ = datasets.make_swiss_roll(n_samples=self.n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        self.points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

    def __len__(self):
        return self.n_samples
       
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        theta = t * 2 * np.pi

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            index = np.random.randint(0, self.n_samples)

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            output[source,0], output[source,1] = np.dot(rotation_matrix, self.points[index])

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        points, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1) 

        points[:,0] = 1.5*(points[:,0] - points[:,0].min()) / (points[:,0].max() - points[:,0].min()) - 0.75
        points[:,1] = 1.5*(points[:,1] - points[:,1].min()) / (points[:,1].max() - points[:,1].min()) - 0.75
        points[:,2] = 1.5*(points[:,2] - points[:,2].min()) / (points[:,2].max() - points[:,2].min()) - 0.75

        points = np.concatenate((points[:,0].reshape(-1,1), points[:,2].reshape(-1,1)), axis=1)

        theta = t * 2 * np.pi

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for i in range(n_samples):

            N_sources = 1

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).
                # Select a section according to the probabilities defined in the paper
        
                index = np.random.randint(0, n_samples)

                samples[i,source,0], samples[i,source,1] = np.dot(rotation_matrix, points[index])

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity
   
# Gaussian circle mixture translation
    
class gaussian_circle_mixture_translation(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(gaussian_circle_mixture_translation, self).__init__()
        self.n_samples = n_samples
        self.sigma=0.1   

        theta_list = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
        circle_radius = 1

        self.mus = np.array([[circle_radius*np.cos(theta), circle_radius*np.sin(theta)] for theta in theta_list])
        self.mus_opposed = np.array([[circle_radius*np.cos(theta+np.pi), circle_radius*np.sin(theta+np.pi)] for theta in theta_list])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)  
        mus_t = np.array([(1-t)*self.mus[i,:] + t*self.mus_opposed[i,:] for i in range(len(self.mus))])

        chosen_mode = np.random.choice(range(8), p=[1/8]*8)

        x, y = np.random.multivariate_normal(mus_t[chosen_mode,:], (self.sigma**2)*np.eye(2))

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        # Define the boundaries of each section 
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        mus_t = np.array([(1-t)*self.mus[i,:] + t*self.mus_opposed[i,:] for i in range(len(self.mus))])

        for i in range(n_samples):
       
            chosen_mode = np.random.choice(range(8), p=[1/8]*8)

            samples[i, 0], samples[i, 1] = np.random.multivariate_normal(mus_t[chosen_mode,:], (self.sigma**2)*np.eye(2))

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_gaussian_circle_mixture_translation(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_gaussian_circle_mixture_translation, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma=0.01   

        theta_list = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
        circle_radius = 1

        self.mus = np.array([[circle_radius*np.cos(theta), circle_radius*np.sin(theta)] for theta in theta_list])
        self.mus_opposed = np.array([[circle_radius*np.cos(theta+np.pi), circle_radius*np.sin(theta+np.pi)] for theta in theta_list])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))
        mus_t = np.array([(1-t)*self.mus[i,:] + t*self.mus_opposed[i,:] for i in range(len(self.mus))])

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True   

            chosen_mode = np.random.choice(range(8), p=[1/8]*8)

            output[source,0], output[source,1] = np.random.multivariate_normal(mus_t[chosen_mode,:], (self.sigma**2)*np.eye(2))

        return np.array([t]), output, mask_activity
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)

    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):

        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere
        t = t.numpy()
        mus_t = np.array([(1-t)*self.mus[i,:] + t*self.mus_opposed[i,:] for i in range(len(self.mus))])

        for i in range(n_samples):

            N_sources = 1

            for source in range(N_sources):
                mask_activity[i,source] = True

                chosen_mode = np.random.choice(range(8), p=[1/8]*8)

                samples[i, source, 0], samples[i, source, 1] = np.random.multivariate_normal(mus_t[chosen_mode,:], (self.sigma**2)*np.eye(2))

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Single Gaussian not centered

class single_gauss_not_centered(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(single_gauss_not_centered, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.2   
        self.mean1 = np.array([0.25,0.25])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

        # Sample a point uniformly from the selected section
        x,y = np.random.multivariate_normal(self.mean1, np.eye(2)*self.sigma1**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            x,y = np.random.multivariate_normal(self.mean1, np.eye(2)*self.sigma1**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_gauss_not_centered(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_single_gauss_not_centered, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.2   
        self.mean1 = np.array([0.25,0.25])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            x,y = np.random.multivariate_normal(self.mean1, np.eye(2)*self.sigma1**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                x,y = np.random.multivariate_normal(self.mean1, np.eye(2)*self.sigma1**2)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Sinusoidal Gaussian

class single_gauss_sinusoidal(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(single_gauss_sinusoidal, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.075  

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        mean1 = np.array([(t-1/2),(1/2)*np.sin(10*(t-1/2))])

        x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        mean1 = np.array([(t-1/2),(1/2)*np.sin(10*(t-1/2))])

        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_gauss_sinusoidal(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_single_gauss_sinusoidal, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        # self.sigma1=0.1 
        self.sigma1 = 0.075  
        # self.mean1 = np.array([0.25,0.25])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))
        # mean1 = np.array([t/2,np.sin(10*t)])
        mean1 = np.array([(t-1/2),(1/2)*np.sin(10*(t-1/2))])

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere
        # mean1 = np.array([t/2,np.sin(10*t)])
        mean1 = np.array([(t-1/2),(1/2)*np.sin(10*(t-1/2))])

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Sinusoidal Imbalanced Gaussian

class single_gauss_imbalanced_sinusoidal(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super(single_gauss_imbalanced_sinusoidal, self).__init__()
        self.n_samples = n_samples
        self.sigma1=0.075  
        # self.mean1 = np.array([0.25,0.25])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = self.imbalanced_input_dist()

        # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

        mean1 = np.array([(t-1/2),(1/2)*np.sin(12*(t-1/2))])

        # Sample a point uniformly from the selected section
        x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

        return torch.Tensor([t, x, y])

    def imbalanced_input_dist(self) :

        hidden_part_min = 0.5
        hidden_part_max = 0.8

        # We sample from the hidden part with a probability of 1/100
        hidden_part_prob = 1/10000

        region_choice = np.random.choice([0,1], p=[hidden_part_prob, 1-hidden_part_prob])

        if region_choice == 0 :
            t = np.random.uniform(hidden_part_min, hidden_part_max)

        else :
            length = hidden_part_max-hidden_part_min
            choice_uniform_part = np.random.choice([0,1],p=[hidden_part_min/(1-length), (1-hidden_part_max)/(1-length)])

            if choice_uniform_part == 0 :

                t = np.random.uniform(0, hidden_part_min)

            else :

                t = np.random.uniform(hidden_part_max, 1)

        return t 

    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        mean1 = np.array([(t-1/2),(1/2)*np.sin(12*(t-1/2))])

        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_gauss_imbalanced_sinusoidal(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super(MultiSources_single_gauss_imbalanced_sinusoidal, self).__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1 = 0.075  

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = self.imbalanced_input_dist()
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))
        mean1 = np.array([(t-1/2),(1/2)*np.sin(12*(t-1/2))])

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True

            x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity

    def imbalanced_input_dist(self) :

        hidden_part_min = 0.5
        hidden_part_max = 0.8

        # We sample from the hidden part with a probability of 1/100
        hidden_part_prob = 1/10000

        region_choice = np.random.choice([0,1], p=[hidden_part_prob, 1-hidden_part_prob])

        if region_choice == 0 :
            t = np.random.uniform(hidden_part_min, hidden_part_max)

        else :
            length = hidden_part_max-hidden_part_min
            choice_uniform_part = np.random.choice([0,1],p=[hidden_part_min/(1-length), (1-hidden_part_max)/(1-length)])

            if choice_uniform_part == 0 :

                t = np.random.uniform(0, hidden_part_min)

            else :

                t = np.random.uniform(hidden_part_max, 1)

        return t 
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere
        # mean1 = np.array([t/2,np.sin(10*t)])
        mean1 = np.array([(t-1/2),(1/2)*np.sin(12*(t-1/2))])

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                x,y = np.random.multivariate_normal(mean1, np.eye(2)*self.sigma1**2)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Fixed mixture of gaussians
    
class mixture_different_sizes(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.26169342,  0.95915058],
        [ 0.48390997,  0.20995194],
        [-0.71750153, -0.73206697],
        [-0.92178159,  0.77924765],
        [ 0.2109131 ,  0.44279254],
        [-1.        ,  1.        ],
        [ 0.69343323, -0.61216186],
        [-0.6636728 , -0.67373665],
        [-0.40832588,  0.05268336],
        [-0.14195407, -0.44427853],
        [ 0.23331097, -0.76718149],
        [-0.43355993, -0.28439105],
        [-0.09163245,  0.60687377],
        [-0.62644244,  0.03029185],
        [ 0.19276508, -0.96518425],
        [ 0.22432493, -0.70114699],
        [-0.90724727,  0.95525883],
        [ 0.97124942,  0.65629045],
        [-0.40755092, -0.856181  ],
        [ 0.38428675, -0.12735955]])

        self.sigmas_list = 0.1*np.ones((20,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        self.modes_probs = 20*[1/20]

        # self.modes_list = [0,1,2,3,4,5,6,7,8,9]
        # self.modes_probs = [3/20,1/20,3/20,3/20,3/20,3/20,1/20,1/20,1/20,1/20]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode
        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_mixture_different_sizes(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.26169342,  0.95915058],
        [ 0.48390997,  0.20995194],
        [-0.71750153, -0.73206697],
        [-0.92178159,  0.77924765],
        [ 0.2109131 ,  0.44279254],
        [-1.        ,  1.        ],
        [ 0.69343323, -0.61216186],
        [-0.6636728 , -0.67373665],
        [-0.40832588,  0.05268336],
        [-0.14195407, -0.44427853],
        [ 0.23331097, -0.76718149],
        [-0.43355993, -0.28439105],
        [-0.09163245,  0.60687377],
        [-0.62644244,  0.03029185],
        [ 0.19276508, -0.96518425],
        [ 0.22432493, -0.70114699],
        [-0.90724727,  0.95525883],
        [ 0.97124942,  0.65629045],
        [-0.40755092, -0.856181  ],
        [ 0.38428675, -0.12735955]])

        self.sigmas_list = 0.1*np.ones((20,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        self.modes_probs = 20*[1/20]

        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Diracs rectangle 
    
class diracs_rectangle(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        # self.means_list = np.array([[-0.75,  -0.1],
        # [ -0.75,  0.1],
        # [0.75, -0.1],
        # [0.75,  0.1]])

        self.means_list = np.array([[-0.75,  -0.1],
        [ -0.75,  0.1],
        [-0.9,  -0.1],
        [ -0.9,  0.1],
        ])

        self.sigmas_list = 0.01*np.ones((4,1))

        self.modes_list = [0,1,2,3]
        self.modes_probs = 4*[1/4]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_diracs_rectangle(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.75,  -0.1],
        [ -0.75,  0.1],
        [-0.9,  -0.1],
        [ -0.9,  0.1],
        ])

        self.sigmas_list = 0.01*np.ones((4,1))

        self.modes_list = [0,1,2,3]
        self.modes_probs = 4*[1/4]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Single 3D Gaussian not centered

class single_3D_gauss_not_centered(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples
        self.sigma1=0.2   
        self.mean1 = np.array(3*[0.25])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # Sample a point uniformly from the selected section
        x,y,z = np.random.multivariate_normal(self.mean1, np.eye(3)*self.sigma1**2)

        return torch.Tensor([t, x, y, z])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 3))
        for i in range(n_samples):

            samples[i, 0],samples[i, 1],samples[i, 2] = np.random.multivariate_normal(self.mean1, np.eye(3)*self.sigma1**2)

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_3D_gauss_not_centered(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.2   
        self.mean1 = np.array(3*[0.25])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,3))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            output[source,0], output[source,1],output[source,2] = np.random.multivariate_normal(self.mean1, np.eye(3)*self.sigma1**2)

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 3))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,3))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                output[source,0], output[source,1], output[source,2] = np.random.multivariate_normal(self.mean1, np.eye(3)*self.sigma1**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]
                samples[i,source,2] = output[source,2]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity


# Diracs rectangle and circle
    
class diracs_rectangle_and_circle(data.Dataset):
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        # self.means_list = np.array([[-0.75,  -0.1],
        # [ -0.75,  0.1],
        # [0.75, -0.1],
        # [0.75,  0.1]])

        self.means_list = np.array([[0.75,  -0.70],
                                    [ 0.75,  -0.85],
                                    [0.9,  -0.70],
                                    [0.9,  -0.85],
                                    [-0.75,  -0.1],
                                    [ -0.75,  0.1],
                                    [-0.9,  -0.1],
                                    [ -0.9,  0.1],
                                    [0.75,0.8],
                                    [0.80,0.90],
                                    [0.,0.]
                                    ])


        self.sigmas_list = np.concatenate((0.01*np.ones((10,1)),0.05*np.ones((1,1))), axis=0)

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10]
        self.modes_probs = 11*[1/11]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_diracs_rectangle_and_circle(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[0.75,  -0.70],
                                    [ 0.75,  -0.85],
                                    [0.9,  -0.70],
                                    [0.9,  -0.85],
                                    [-0.75,  -0.1],
                                    [ -0.75,  0.1],
                                    [-0.9,  -0.1],
                                    [ -0.9,  0.1],
                                    [0.75,0.8],
                                    [0.80,0.90],
                                    [0.,0.]
                                    ])


        self.sigmas_list = np.concatenate((0.01*np.ones((10,1)),0.05*np.ones((1,1))), axis=0)

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10]
        self.modes_probs = 11*[1/11]


    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity


# Diracs cross
    
class diracs_cross(data.Dataset):
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.77777778, -0.77777778],
                                    [-0.55555556, -0.55555556],
                                    [-0.33333333, -0.33333333],
                                    [-0.11111111, -0.11111111],
                                    [ 0.11111111,  0.11111111],
                                    [ 0.33333333,  0.33333333],
                                    [ 0.55555556,  0.55555556],
                                    [ 0.77777778,  0.77777778],
                                    [ 0.77777778, -0.77777778],
                                    [ 0.55555556, -0.55555556],
                                    [ 0.33333333, -0.33333333],
                                    [ 0.11111111, -0.11111111],
                                    [-0.11111111,  0.11111111],
                                    [-0.33333333,  0.33333333],
                                    [-0.55555556,  0.55555556],
                                    [-0.77777778, 0.77777778]
                                    ])


        self.sigmas_list = 0.01*np.ones((16,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.modes_probs = 16*[1/16]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_diracs_cross(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.77777778, -0.77777778],
                                    [-0.55555556, -0.55555556],
                                    [-0.33333333, -0.33333333],
                                    [-0.11111111, -0.11111111],
                                    [ 0.11111111,  0.11111111],
                                    [ 0.33333333,  0.33333333],
                                    [ 0.55555556,  0.55555556],
                                    [ 0.77777778,  0.77777778],
                                    [ 0.77777778, -0.77777778],
                                    [ 0.55555556, -0.55555556],
                                    [ 0.33333333, -0.33333333],
                                    [ 0.11111111, -0.11111111],
                                    [-0.11111111,  0.11111111],
                                    [-0.33333333,  0.33333333],
                                    [-0.55555556,  0.55555556],
                                    [-0.77777778, 0.77777778]
                                    ])


        self.sigmas_list = 0.01*np.ones((16,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.modes_probs = 16*[1/16]


    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity


# Diracs cercle two layers
    
class dirac_cercle_two_layers(data.Dataset):
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        # self.means_list = np.array([[-0.75,  -0.1],
        # [ -0.75,  0.1],
        # [0.75, -0.1],
        # [0.75,  0.1]])

        self.means_list = np.array([[ 8.00000000e-01,  0.00000000e+00],
                                    [ 4.00000000e-01,  6.92820323e-01],
                                    [-4.00000000e-01,  6.92820323e-01],
                                    [-8.00000000e-01,  9.79717439e-17],
                                    [-4.00000000e-01, -6.92820323e-01],
                                    [ 4.00000000e-01, -6.92820323e-01],
                                    [-4.00000000e-01, -1.60000000e-01],
                                    [-1.33333333e-01, -1.60000000e-01],
                                    [ 1.33333333e-01, -1.60000000e-01],
                                    [-4.00000000e-01, -1.33333333e-01],
                                    [-1.33333333e-01, -1.33333333e-01],
                                    [ 1.33333333e-01, -1.33333333e-01],
                                    [-4.00000000e-01, -1.06666667e-01],
                                    [-1.33333333e-01, -1.06666667e-01],
                                    [ 1.33333333e-01, -1.06666667e-01],
                                    ])


        self.sigmas_list = 0.01*np.ones((15,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.modes_probs = 15*[1/15]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_dirac_cercle_two_layers(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[ 8.00000000e-01,  0.00000000e+00],
                                    [ 4.00000000e-01,  6.92820323e-01],
                                    [-4.00000000e-01,  6.92820323e-01],
                                    [-8.00000000e-01,  9.79717439e-17],
                                    [-4.00000000e-01, -6.92820323e-01],
                                    [ 4.00000000e-01, -6.92820323e-01],
                                    [-4.00000000e-01, -1.60000000e-01],
                                    [-1.33333333e-01, -1.60000000e-01],
                                    [ 1.33333333e-01, -1.60000000e-01],
                                    [-4.00000000e-01, -1.33333333e-01],
                                    [-1.33333333e-01, -1.33333333e-01],
                                    [ 1.33333333e-01, -1.33333333e-01],
                                    [-4.00000000e-01, -1.06666667e-01],
                                    [-1.33333333e-01, -1.06666667e-01],
                                    [ 1.33333333e-01, -1.06666667e-01],
                                    ])


        self.sigmas_list = 0.01*np.ones((15,1))

        self.modes_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.modes_probs = 15*[1/15]


    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Three gaussians
    
class three_gaussians(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.sigmas_list = 0.1*np.ones((3,1))

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_three_gaussians(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.sigmas_list = 0.1*np.ones((3,1))

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Smaller three gaussians
    
class smaller_three_gaussians(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.sigmas_list = 0.1*np.ones((3,1))

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        print(self.means_list[choice_mode,:])

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_smaller_three_gaussians(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.sigmas_list = 0.1*np.ones((3,1))

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Three gaussians changing variance
    
class three_gaussians_changevar(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        self.sigmas_list = self.function_t(t)*np.ones((3,1))

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            self.sigmas_list = self.function_t(t)*np.ones((3,1))

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

    def function_t(self,t):
        return t*0.3
    
class MultiSources_three_gaussians_changevar(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.5,  -0.5],
        [0.,  0.5],
        [0.5,  -0.5],
        ])

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            self.sigmas_list = self.function_t(t)*np.ones((3,1))

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

                self.sigmas_list = self.function_t(t)*np.ones((3,1))
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

    def function_t(self,t):
        return t*0.3

# Three gaussians changing distance
    
class three_gaussians_changedist(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        # self.means_list = np.array([[-0.5,  -0.5],
        # [0.,  0.5],
        # [0.5,  -0.5],
        # ])

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode

        self.means_list = self.mean_list_t(t)
        self.sigmas_list = self.function_t(t)*np.ones((3,1))

        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            self.means_list = self.mean_list_t(t)
            self.sigmas_list = self.function_t(t)*np.ones((3,1))
        
            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples

    def function_t(self,t):
        return 0.1

    def mean_list_t(self,t):

        return np.array([[t*(-0.5),  t*(-0.5)],
        [0,  t*0.5],
        [t*0.5,  t*-0.5],
        ])
    
class MultiSources_three_gaussians_changedist(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        # self.means_list = np.array([[-0.5,  -0.5],
        # [0.,  0.5],
        # [0.5,  -0.5],
        # ])

        self.modes_list = [0,1,2]
        self.modes_probs = 3*[1/3]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            self.means_list = self.mean_list_t(t)
            self.sigmas_list = self.function_t(t)*np.ones((3,1))

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

                self.means_list = self.mean_list_t(t)
                self.sigmas_list = self.function_t(t)*np.ones((3,1))
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

    def function_t(self,t):
        return t*0.3

    def mean_list_t(self,t):

        return np.array([[t*(-0.5),  t*(-0.5)],
        [0,  t*0.5],
        [t*0.5,  t*-0.5],
        ])

# Fixed mixture of gaussians annealing
    
class annealing_mixture_different_sizes(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[ 0.06877488, -0.66161435],
       [ 0.16781071, -0.41815127],
       [ 0.30811146, -0.20357364],
       [ 0.52268909,  0.08115436],
       [ 0.23796108,  0.15130474],
       [-0.40989829,  0.15543123]])

        self.sigmas_list = 0.1*np.ones((6,1))

        self.modes_list = [0,1,2,3,4,5]
        self.modes_probs = 6*[1/6]

        # self.modes_list = [0,1,2,3,4,5,6,7,8,9]
        # self.modes_probs = [3/20,1/20,3/20,3/20,3/20,3/20,1/20,1/20,1/20,1/20]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode
        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_annealing_mixture_different_sizes(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[ 0.06877488, -0.66161435],
       [ 0.16781071, -0.41815127],
       [ 0.30811146, -0.20357364],
       [ 0.52268909,  0.08115436],
       [ 0.23796108,  0.15130474],
       [-0.40989829,  0.15543123]])

        self.sigmas_list = 0.1*np.ones((6,1))
        self.modes_list = [0,1,2,3,4,5]
        self.modes_probs = 6*[1/6]

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], np.eye(2)*self.sigmas_list[choice_mode,:]**2)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Single Gaussian ellongated not centered

class single_ellongated_gauss(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples
        self.sigma1=0.2   
        self.sigma2=0.1
        self.mean = np.array([0.,0.])
        self.covariance_matrix = np.array([[self.sigma1**2, 0], [0, self.sigma2**2]])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        t = np.random.uniform(0, 1)

        # section = np.random.choice([1, 2, 3, 4], p=[p1, p2, p3, p4])

        # Sample a point uniformly from the selected section
        x,y = np.random.multivariate_normal(self.mean, self.covariance_matrix)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            x,y = np.random.multivariate_normal(self.mean, self.covariance_matrix)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_single_ellongated_gauss(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t
        self.sigma1=0.2   
        self.sigma2=0.1
        self.mean = np.array([0.,0.])
        self.covariance_matrix = np.array([[self.sigma1**2, 0], [0, self.sigma2**2]])

        self.grid_t = grid_t

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_distribution = np.random.choice([1,2],p=[(1 - t), t])

            x,y = np.random.multivariate_normal(self.mean, self.covariance_matrix)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                x,y = np.random.multivariate_normal(self.mean, self.covariance_matrix)

                output[source,0], output[source,1] = x,y

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

# Fixed mixture of gaussians annealing
    
class mixture_ellongated_gauss(data.Dataset):
    """Class for generating the initial version of the "toy" dataset, proposed by Rupperecht et al. in 
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Rupprecht_Learning_in_an_ICCV_2017_paper.pdf."""
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

        self.means_list = np.array([[-0.5,  0.5],
        [ 0.5,  0.5]])

        self.sigma1=0.1   
        self.sigma2=0.2
        self.mean = np.array([0.,0.])
        self.covariance_matrix = np.array([[self.sigma1**2, 0], [0, self.sigma2**2]])

        self.modes_list = [0,1]
        self.modes_probs = 2*[1/2]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Sample a value of t uniformly
        # from [0, 1]
        t = np.random.uniform(0, 1)
   
        choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
        # sample from a gaussian mixture given the selected mode
        x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], self.covariance_matrix)

        return torch.Tensor([t, x, y])
    
    def generate_dataset_distribution(self, t, n_samples,plot=False):
        """Generate n_samples samples from the dataset distribution for a given value of t."""
        
        # Generate n_samples samples
        samples = np.zeros((n_samples, 2))
        for i in range(n_samples):

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)

            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], self.covariance_matrix)

            samples[i, 0] = x
            samples[i, 1] = y

        if plot : 
            plt.scatter(samples[:, 0], samples[:, 1], s=5)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the toy dataset with t={}'.format(n_samples, t))
            plt.show()
            
        return samples
    
class MultiSources_mixture_ellongated_gauss(data.Dataset):

    """Class for generating the proposed variant of the dataset."""
    def __init__(self, n_samples=100, Max_sources=2,grid_t=False,t=None) :
        super().__init__()
        self.n_samples = n_samples
        self.Max_sources = Max_sources
        self.t = t

        self.grid_t = grid_t
        self.means_list = np.array([[-0.5,  0.5],
        [ 0.5,  0.5]])

        self.sigma1=0.1  
        self.sigma2=0.2
        self.mean = np.array([0.,0.])
        self.covariance_matrix = np.array([[self.sigma1**2, 0], [0, self.sigma2**2]])

        self.modes_list = [0,1]
        self.modes_probs = 2*[1/2]

        # if self.grid_t is True and self.t is None : # At evaluation time, we evaluate on a grid of t values.
            # self.t_grid = torch.linspace(0.0, 1.0, steps=n_samples)
        # self.t_grid = np.linspace(0.0, 1.0, num=n_samples)

    def __len__(self):
        return self.n_samples
    
    def define_t_grid(self) :
        self.t_grid = np.linspace(0.0, 1.0, num=self.n_samples)
    
    def __getitem__(self, index):
        # Sample a value of t uniformly from [0, 1]
        if self.grid_t is True : # At evaluation time, the t values are sampled on a grid.
            t = self.t_grid[index % len(self.t_grid)]
        elif self.t is None : # At training time, the t values are sampled uniformly if the t value is not specificed. 
            t = np.random.uniform(0, 1)
        else: 
            t = self.t
        mask_activity = np.zeros((self.Max_sources,1)) # True if the target is active, False otherwise
        mask_activity = mask_activity > 0 # False everywhere

        # Sample the number of sources
        N_sources = 1

        output = np.zeros((self.Max_sources,2))

        # Sample the position of the sources given the number of sources. 
        for source in range(N_sources):
            mask_activity[source,0] = True
            # Select a section according to the probabilities defined in the paper

            choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
            # sample from a gaussian mixture given the selected mode
            x,y = np.random.multivariate_normal(self.means_list[choice_mode,:], self.covariance_matrix)

            output[source,0], output[source,1] = x,y

        return np.array([t]), output, mask_activity
    
    def generate_dataset_distribution(self, t, n_samples,plot_one_sample=False,Max_sources=2):
        """Generate a dataset with a fixed value of t."""        
        # Generate n_samples samples
        samples = np.zeros((n_samples, Max_sources, 2))
        mask_activity = np.zeros((n_samples, Max_sources))
        mask_activity = mask_activity > 0 # False everywhere

        for i in range(n_samples):

            N_sources = 1

            output = np.zeros((N_sources,2))

            for source in range(N_sources):
                mask_activity[i,source] = True # This mask stands for the activity of the target (for handling multiple targets).

                choice_mode = np.random.choice(self.modes_list,p=self.modes_probs)
        
                # sample from a gaussian mixture given the selected mode
                output[source,0], output[source,1] = np.random.multivariate_normal(self.means_list[choice_mode,:], self.covariance_matrix)

                samples[i,source,0] = output[source,0]
                samples[i,source,1] = output[source,1]

        if plot_one_sample : 
            plt.scatter(samples[0,:, 0][mask_activity[0,:]], samples[0,:, 1][mask_activity[0,:]], marker='*', c='red', s=100)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.title('{} samples of the multi-source toy dataset with t={}'.format(1, t))
            plt.show()
            
        return samples, mask_activity

