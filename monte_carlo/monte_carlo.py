import numpy as np
import matplotlib.pyplot as plt
import warnings

from mfa_ua.monte_carlo.sampling import Sampler

class MonteCarlo:
    '''
    For conducting an MC analysis, storing the results and visualizing them.
    
    Attributes:
    - function: the function that is executed each iteration of the MC simulation
    - sample: the Sampler object that provides the sets of inputs
    - parameters: a list of parameters taken from the sample
    - iterations: number of function executions and results
    - result_lists: lists of each paramater with results from all runs
    - results_sets: list of sets of all function outputs
    - figures: visualisations of each result 
    ______________________less important________________________________
    - no_parameters:
    - no_bins:
    - visualisations: True/False
    - result_names: names of the results from the function
    - results_order: dict for easier retrieval of the results
    - results_types: scalar or timeseries and so on


    Methods: 
    - analze: prepares everything and runs a MC simulation.
    Optionally one can set iterations and bins for histograms; 
    visualizations can be made and shown with the respective keywords.
    - histogram: histogram of named scalar result.
    - plot_timeseries: plots uncertainty on a timeseries result.
    '''
    def __init__(self, function, sample: Sampler , iterations: int = 10000, no_bins: int = 100, 
                 probablistic_model = False, visualisations = False) -> None:
        '''
        Arguments:
        - function: must give two outputs,  one vetor of objects for the 
        result(s) and one vector with the name(s) of these; it takes a 
        list of inputs in the order of parameters
        - sample: a Sampler object with a list of parameters which will 
        make sets of function inputs 
        - iterations: number of MC iterations, default is 10k
        - no_bins: number of bins for histograms, default is 100.
        - probablistic_model: not yet supported, but if the model itself 
        contains random processes we could just reuse a single set of 
        parameters multiple times.
        '''        
        self.function = function
        self.sample = sample
        self.parameters = sample.parameters
        self.no_parameters = len(self.parameters)
        self.iterations = iterations
        self.no_bins = no_bins
        self.visualisations = visualisations
        

    
    def analyze(self, iterations:int = None, no_bins:int = None, visualisations:bool =False)->None:
        '''
        The main method of the MC class - conducts a full MonteCarlo 
        analysis and handles the results.

        Arguments:
        - iterations: if one wants to change the iterations from the 
        number specified in the init (will overwrite init)
        - no_bins: for the histograms (will overwrite init)
        - visialisations: whether or not the results will be plotted.
        '''
        #prepare everything
        #use default iterations and bins if not specified
        if not iterations:
            iterations = self.iterations
        self.iterations = iterations
        if not visualisations:
            visualisations = self.visualisations
        self.visualisations = visualisations
        if not no_bins:
            no_bins = 'auto'
        self.no_bins = no_bins
        self._check_iterations()
        #self._set_number_of_bins()
        #TODO improve on no bins

        
        #calculating the outputs (this takes time!)
        self.result_sets,self.l_r_names = zip(*tqdm([self.function(inputs) for inputs in 
                                                self.sample.parameter_sets[0:self.iterations]]))
        self.result_names = self.l_r_names[0]
        
        #check if the same names are returned each time
        for index, current in enumerate(self.l_r_names, start = 1):
            prior = self.l_r_names[index-1]
            if prior!= current:
                raise Exception(f'Your function does not always return the same names: in result '\
                                f'{index} the names were {current}, before they were {prior}.')

        #repackage results
        self.result_lists = []
        for output in range(len(self.result_names)):
            self.result_lists.append([set[output] for set in self.result_sets])

        #establish order of the results:
        self.results_order = {}
        self.results_type = {}
        for index, (result_name, result_example) in enumerate(zip(self.result_names, self.result_sets[0])):
            self.results_order[result_name] = index
            if type(result_example) == list:
                if len(result_example) == 2: # if time & results are specified
                    self.results_type[result_name] = 'timeseries'
                else:
                    self.results_type[result_name] = 'not supported list'
                    warnings.warn(f'Your result {result_name} is a not identifiable list.')
            elif type(result_example) in [int, float, np.float64]:
                self.results_type[result_name] = 'scalar'
            else:
                self.results_type[result_name] = f'type(result_example)'
                warnings.warn(f'the type of your result {result_name} is {type(result_example)},'\
                              f'which means we cannnot plot it.')
        #make and show plots as wished:
        if self.visualisations:
            self.figures= self._plot_results(show = True)
        return
    

    
    def histogram(self, result_name:str, colour:str = 'royalblue')->plt.figure:
        '''A simple histogram for the scalar results of the MC'''
        #prepare the data
        index = self.results_order[result_name]
        results = self.result_lists[index]
        #do statistics:
        mean = np.mean(results)
        std = np.std(results)

        fig = plt.figure(figsize=(6,4.5))
        n,_,_ = plt.hist(results, bins = self.no_bins, color = colour, alpha = 0.8)
        y_coord = [0,max(n)] #for the stat intervals
        plt.plot([mean, mean], y_coord, c = 'black', lw = 2, label = 'mean', alpha = 0.8)
        plt.plot([mean+std, mean+std], y_coord,'--',c = 'black', lw = 2, alpha = 0.6)
        plt.plot([mean-std, mean-std], y_coord,'--', c = 'black', lw = 2, alpha = 0.6, 
                 label = '1 std - 68%')
        plt.plot([mean+2*std, mean+2*std], y_coord,ls = ':',c = 'black', lw = 2, alpha = 0.4)
        plt.plot([mean-2*std, mean-2*std], y_coord,ls = ':', c = 'black', lw = 2, alpha = 0.4,
                 label = '2 stds - 95%')
        plt.legend(loc = 'best')
        plt.xlabel(f'Values of {result_name}')
        plt.ylabel('Number of runs')
        plt.title(f'{result_name} results from MC simulation with {self.iterations} iterations')
        plt.close()
        return fig

    def plot_timeseries(self, result_name:str,  colour:str = 'darkorange', scatter:bool = False,
                        y_unit = 'unknown', time_unit = 'unknown', show:bool = True) -> plt.figure:
        '''Plots timeseries with scatter plot and CI intervals.'''
        index = self.results_order[result_name]
        results = self.result_lists[index]
        result_by_year = [[] for _ in results[0][0]]
        for result in results:
            for index, year_value in enumerate(result[1]):
                result_by_year[index].append(year_value) 
        means = [np.mean(year) for year in result_by_year]
        stds = [np.std(year) for year in result_by_year]
        t = results[0][0]

        fig = plt.figure(figsize = (20,10))
        plt.fill_between(x = t, y1 = np.subtract(means, np.einsum('i,->i',stds,2)), 
                         y2 = np.subtract(means, np.einsum('i,->i',stds,1)), alpha = 0.25, 
                         color = colour, label = '95% interval')
        plt.fill_between(x = t, y1 = np.add(means, np.einsum('i,->i',stds,2)),
                         y2 = np.add(means, np.einsum('i,->i',stds,1)), alpha = 0.25, color = colour)
        plt.fill_between(x = t, y1 = np.subtract(means, np.einsum('i,->i',stds, 1)), 
                         y2 = np.add(means, np.einsum('i,->i',stds,1)), alpha = 0.55, color = colour,
                         label = '68% interval')
        plt.plot(t, means, color = 'black', label = f'mean value for {result_name} ')
        if scatter:
            for xe, ye in zip(t, result_by_year):
                plt.scatter([xe] * len(ye), ye, color = 'black', s = 10**3/len(ye), alpha = 0.1)
            #plt.scatter(x = t, y = result_by_year, alpha = 0.1)
        plt.xlabel(f'Time in {time_unit} ')
        plt.ylabel(f'{result_name} in {y_unit}.')
        plt.legend(loc = 'best')
        plt.title(f'{result_name} over time')
        plt.close()
        if not show:
            plt.close()
        return fig

    def _plot_results(self, show:bool = False) -> List[plt.figure]:
        '''Plots all results from the MC function that can be plotted.'''
        figures = []
        #checks which results are timeseries to plot histograms or other plots.
        for name in self.result_names: 
            if self.results_type[name] == 'scalar':
                fig = self.histogram(result_name = name)
            elif self.results_type[name] == 'timeseries':
                fig = self.plot_timeseries(result_name= name)
            else:
                warnings.warn(f'The type of the result {name} ({self.results_type[name]}) cannot '\
                              f'be plotted here yet.')
                break
            figures.append(fig)
        if show:
            for fig in figures:
                if fig: #if figure is not None
                    show_figure(fig)
        return figures 

    def _set_number_of_bins(self) -> None:
        '''
        number of bins is 100 per default, except no_iterations < 100, 
        then we use 1/10 times of the operations (but at least one).
        '''
        if(self.no_bins<self.iterations/10):
            self.no_bins = int(np.ceil(self.no_bins))
        else:
            self.no_bins = int(np.ceil(self.iterations/10))
        return
    
    def _check_iterations(self) -> None:
        '''Warns user of unreasonable inputs and ensures ok samples.'''
        if self.iterations < 1:
            raise Exception(f'You cannot have {self.iterations} iterations - use pos. integer!')
        elif self.iterations < 100:
            warnings.warn(f"You use {self.iterations} iterations, that's probably not enough.")
        if self.iterations < self.sample.samplesize:
            warnings.warn(f'You want to do an MC with {self.iterations} iterations, but sampled'\
                          f' more sets of inputs ({self.sample.samplesize}).')

        if self.iterations > self.sample.samplesize:
            mode = self.sample.mode
            _ = self.sample.sample(mode = mode, samplesize = self.iterations)
            warnings.warn(f'You picked more MC iterations than there were samples availabale, '\
                          f'so we resampled with {self.iterations} in {mode} mode.')
