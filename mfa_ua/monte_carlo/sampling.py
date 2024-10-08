import matplotlib.pyplot as plt

from mfa_ua.monte_carlo.parameters import ScalarParameter

class Sampler:
    '''
    Collection of probabilistically determined parameters for sampling 
    in sets. Is used for the MC class and generates specific samples.

    Attributes:
    - parameters: list of parameters with a sampling method.
    - parameters_order: dict that maps names and shortnames with 
      the index in the parameters list (automatically).
    - samplesize: number of sample sets from most recent sampling.
    - mode: mode for sampling from most recent sampling.
    - parameter_samples: list (one entry per parameter) of lists of
      samples of that parameter
    - parameter_sets: samplesize sets (as a list) of all parameters.

    Methods:
    - init: takes list of parameters and prepares for the sample method.
    - sample: returns parameter_sets for a given mode (MonteCarlo (MC), 
      later also LatinHypercube (LH) or correlated_MC) and samplesize.
    - plot2D: scatterplot of parameter_sets for 2 parameters' values.
    - plot3D: 3Dscatterplot of parameter_sets for 3 parameters' values.
    - private _check_samples_for_plot: makes sure we sampled before 
    plotting (and that types are ok) and sends warning if we didn't.
    - private _repackage: builds a list of lists (sets of parameter 
      samples) from list of parameter samples -effectively it 
      translates parameter_samples to parameter_sets
    '''

    figsize = (10,10) # default figure size 
    parameter_samples = None #for the __check samples method
    samplesize = 0
    mode = 'MC' #default mode is MC (random)
    def __init__(self, parameters: list[ScalarParameter] )->None:
        self.parameters = parameters
        self.dimension = len(self.parameters)
        self.parameters_order = {}
        self.parameters_type = {}
        for index, parameter in enumerate(parameters):
            self.parameters_order[parameter.name] = index
            self.parameters_order[parameter.short_name] = index

            if type(parameter) == ScalarParameter:
                self.parameters_type[parameter.name] = 'scalar'
                self.parameters_type[parameter.short_name] =  'scalar'
            elif type(parameter) == TimeseriesParameter:
                self.parameters_type[parameter.name] = 'timeseries'
                self.parameters_type[parameter.short_name] =  'timeseries'
            elif type(parameter) == ConstantParameter:
                self.parameters_type[parameter.name] = 'constant'
                self.parameters_type[parameter.short_name] =  'constant'
            else:
                warnings.warn(f'Your parameter {parameter.name} is of type {type(parameter)}, '\
                              f'which is not supported. If it has a compatible sampling method, '
                              f'MC base functions might still work.')
        

    def sample(self, mode: str = 'MC', samplesize:int = 1000, correlations: list = None) ->list:
        '''
        Creates list of sets and lists of samples and sets of parameters
        that are given to the Sampler object. 
        Parameter_sets is a list (size samplesize) that contains lists 
        (size no_parameters) of one value of each parameter in order, 
        while parameter samples is a list (size  no_parameters) of lists
        (size samplesize) of values of the parameter
        '''
        self.samplesize = samplesize
        self.mode = mode
       
        if mode == 'MC':
            self.parameter_samples = [para.sampling(samplesize) for para in self.parameters]
            self.parameter_sets = self._repackage(self.parameter_samples)
        elif mode == 'LHS':
            '''
            LHS sampling with scipys LHS sampler. Only works with parameters 
            that support a PPF (percent point function - inverse of cdf)
            '''
            LHS_sampler = sts.qmc.LatinHypercube(d = self.dimension)
            lhs_values = LHS_sampler.random(samplesize)
            self.parameter_samples = []
            for i, para in enumerate(self.parameters):
                if not hasattr(para, 'ppf'):
                    raise TypeError(f'Parameters in sampler need a ppf function ({para.name} hn).')
                self.parameter_samples.append(para.ppf(lhs_values[:,i]))
            self.parameter_sets = self._repackage(self.parameter_samples)
        elif mode == 'correlated_MC':
            #TODO: understand correlated MC sampling
            raise ValueError('Correlated sampling is not yet implemented.')
        return self.parameter_sets
    
    def plot_2D(self, parameter1:str, parameter2:str, show:bool = True) -> plt.figure:
        '''
        Makes a scatterplot of the sample values of two parameters 
        (specified by their full  or short name).
        
        Arguments: 
        - parameter1/2: either full or short name of the parameters
        - show: whether or not to display the plot (closes the canvas)
        '''
        self._check_samples_for_plot(type = 'scalar', paras = [parameter1, parameter2])
        index_x = self.parameters_order[parameter1.name]
        index_y = self.parameters_order[parameter2.name]
        x = self.parameter_samples[index_x]
        y = self.parameter_samples[index_y]
        fig = plt.figure(figsize=self.figsize)
        #if self.mode == 'MC':
        plt.scatter(x,y, s = 10**5/self.samplesize, c = 'crimson', alpha = 0.6)
        plt.xlabel(f'Parameter {parameter1} in {self.parameters[index_x].unit}')
        plt.ylabel(f'Parameter {parameter2} in {self.parameters[index_y].unit}')
        plt.title(f'Scatterplot ({self.samplesize} points) of {parameter1} and {parameter2}.')
        if not show:
            plt.close()
        return fig
    
    def plot_3D(self, parameter1:str, parameter2:str, parameter3:str, c_map:str = 'hot', 
                interactive:bool = False, show:bool = True) -> plt.figure:
        '''
        For 3D scatter plots of some important model parameters.
        Arguments:
        - parameter1-3: names (long or short) of the parameters 
        - c_map: the colormap for the z axis
        - interactive: if in a jupyter notebook, you might want to try
        your luck making this an interactive graphic.
        - show: whether or not to display the plot (closes the canvas)
        '''
        # Use python guide if you want to change it : 
        # https://pythonguides.com/matplotlib-3d-scatter/
        self._check_samples_for_plot(type = 'scalar', paras = [parameter1, parameter2, parameter3])
        index_x = self.parameters_order[parameter1.name]
        index_y = self.parameters_order[parameter2.name]
        index_z = self.parameters_order[parameter3.name]
        x = self.parameter_samples[index_x]
        y = self.parameter_samples[index_y]
        z = self.parameter_samples[index_z]

        if interactive:
            #TODO check if this can be integrated in a useful way
            print(f'currently not implemented')
            return
            warnings.warn('Interactive mode only works within a jupyter notebook (and sometimes).')
            #%matplotlib notebook
            #%matplotlib ipympl
        
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection ="3d")
        color_map = plt.get_cmap(c_map)
        size_dots = 4*10**4/self.samplesize # adjust as needed if it doesn't look nice.
        scatter_plot = ax.scatter3D(x,y,z, c = z, cmap = color_map, s = size_dots, alpha = 0.5)
        plt.colorbar(scatter_plot, shrink = 0.5)
        ax.set_xlabel(f'Parameter {parameter1} in '\
            f'{self.parameters[index_x].unit}', fontweight ='bold')
        ax.set_ylabel(f'Parameter {parameter2} in '\
            f'{self.parameters[index_y].unit}', fontweight ='bold')
        ax.set_zlabel(f'Parameter {parameter3} in '\
            f'{self.parameters[index_z].unit}', fontweight ='bold')
        plt.title(f'3D SCATTER for {parameter1}, {parameter2} and '\
        f'{parameter3}', fontweight='bold', size= 10)
        if not show:
            plt.close()
        return fig
    
    def plot_timeseries_scalar_together(self):
        #TODO think about smart way to do this
        print('Not yet implemented.')
        pass

    def _repackage(self, parameters_individual_samples:list):
        '''
        For an input of p parameters with n samples each: 
        returns n sets with p parameters.
        '''
        sets = []
        for set_index, _ in enumerate(parameters_individual_samples[-1]):
            set = [parameter[set_index] for parameter in parameters_individual_samples]
            sets.append(set)
        return sets
    
    def _check_samples_for_plot(self, type:str = None, paras:list[str] = None):
        '''Ensures parameter samples are available before we plot.'''
        if type is not None:
            for parameter in paras:
                if self.parameters_type[parameter.name] is not type:
                    warnings.warn(f'The parameter {parameter} you try to plot is probably not '\
                                  f'compatible with the plotting method.')
        if not self.parameter_samples:
            warnings.warn('You wanted to plot before creating samples,so we made 1000 MC samples.')
            _ = self.sample()
        return
