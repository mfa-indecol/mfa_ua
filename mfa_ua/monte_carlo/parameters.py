import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from mfa_ua.parameter_estimation.right_skewed_lognorm import RightSkewedLognorm

class UncertainEntity:
    '''
    Parent class for ScalarParamters and ConstantParameters.
   Includes some basic dunder methods for namuing and comparing.
    '''
    
    def __init__(self, name:str, short_name:str):
        self.name = name
        self.short_name = short_name
        
    def sampling(self, samplesize:int):
        '''Necessary function that returns samplesize many samples'''
        return
    
    def __eq__(self, other):
        '''
        We make sure that for the uncertainEntities it is only the names that 
        determine equality. This should help with comparisons later on.
        '''
        return self.name == other.name and self.short_name==other.short_name
    
    def __str__(self):
        '''For printing in f.e. fstrings'''
        return f'{self.name} ({self.short_name})'
    
    def __repr__(self):
        '''For straightup prints'''
        return self.__str__()

class ScalarParameter(UncertainEntity):
    '''
    Distribution of an uncertain scalar (floating point) parameter.
    This class is wrapping a scipy.stats distribution.

    Attributes:
    - name: long (expressive) name of the parameter.
    - short: short (<5 characters) name for plots with many parameters.
    - unit: short unit (is used in plots)
    - distribution: should comply to the scipy.stats library.
    - value1/2/3/4: parameters for the distribution (scipy.stats library).

    Methods:
    - sampling: returns requested semirandom samples as np.array 
    - plot_samples: histogram of samples generated with sampling
    '''

    figsize = (6,4.5) #default figure size for the plots
    # all supported distributions (in the find_sts function) and their 
    # required inputs (must have a value) are listed here for checks
    conditions_dict = {'norm': [True, True, False, False],
                       'truncnorm': [True, True, True, True],
                       'lognorm': [True, True, True, False],
                       'randint': [True, True, False, False],
                       'uniform': [True, True, False, False],
                       'triang': [True, True, True, False],
                       'right skewed': [True, True, True, False]}
    
    def __init__(self, name:str, short_name:str, unit:str, distribution:str, para1:float = None, 
                 para2:float = None, para3:float = None, para4:float = None, 
                 low_lim:float = None, upp_lim:float = None) -> None:
        '''
        Sets attributes for the methods; parameters are 
        None by default, but most distributions require some values. 
        The order in which para1, para2, para3 and para4 should be 
        given is the same of parameters used in scipy.stats - see link.
        https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/stats.html

        Arguments: 
        - name: long (full) name of the parameter.
        - short_name: abreviation/super short name of the parameter
        - unit: very short unit for plots etc.
        - distribution: the distribution for your parameter. You can 
                        easily implement new distribution from scipy by 
                        creating a new elif in sampling.
                        Every distribution function is required to have
                        a pdf, cdf, ppf and rvs function to ensure full 
                        compatibility with the rest of the module.
        - para1/2/3/4: optional parameterss that define the distribution
        - low_lim and upp_lim: for manual truncation of the distribution
        '''
        self.name = name
        self.short_name = short_name
        self.unit = unit
        self.distribution = distribution.lower()
        self.parameter1 = para1
        self.parameter2 = para2
        self.parameter3 = para3
        self.parameter4 = para4
        self.lower_limit = low_lim
        self.upper_limit = upp_lim

        # raise informative errors if inputs are missing
        self._input_check()
        # assign the correct distribution(inputs) from scipy
        self.distribution_function, self.function_inputs = self._find_sts()
        # for convenient use of the class we assign the mean of the distribution
        self.mean = self.distribution_function.mean(**self.function_inputs)
        
    def sampling(self, samplesize:int) -> np.ndarray:
        '''
        Most important method that is used in the Sampler object.
        Creates an array of 'samplesize' samples for the parameter.
        '''
        if not (samplesize > 0):
            raise ValueError(f'Choose a pos. integer instead of {samplesize} as the samplesize')
        #ensure type compatability:
        samplesize = int(samplesize) 
        return self.distribution_function.rvs(**self.function_inputs, size = samplesize)
    
    def sampling(self, samplesize:int) -> np.ndarray:
        '''
        Most important method that is used in the Sampler object.
        Creates an array of 'samplesize' samples for the parameter.
        If limits for the distribution were given, the distribution is truncated
        here. This happens by rescaling the distribution using cdf and ppf - for 
        more details see: 
        https://stackoverflow.com/questions/11491032/truncating-scipy-random-distributions

        Arguments:
        - samplesize: number of samples (points in array) to be returned

        Returns: 
        - samples: 1 dimensional np.ndarray of length samplesize
        '''
        # ensure type compatability:
        samplesize = int(samplesize) 
        if not (samplesize > 0):
            raise ValueError(f'Choose a pos. integer instead of {samplesize} as the samplesize')
        
        # if limits are given, we will set remaining limits automatically and 
        # sample with np.random and the cdf and ppf.
        if self.lower_limit is not None or self.upper_limit is not None:
            #set either that is None to infinite:
            if self.lower_limit is None:
                self.lower_limit = -10**10
                if self.distribution_function.pdf(self.lower_limit, **self.function_inputs)>10**-6:
                    raise ValueError(f'automatic lower limit not sufficiently low - please reset '\
                                     f'this in the UA code.')
            if self.upper_limit is None:
                self.upper_limit = 10**10
                if self.distribution_function.pdf(self.upper_limit, **self.function_inputs)>10**-6:
                    ValueError(f'automatic upper limit not sufficiently high - please reset this '\
                               f'in the UA code.')
            if not self.lower_limit< self.upper_limit:
                raise ValueError(f'Lower limit must be below upper limit.')
            
            # create truncation normalisation
            nrm = self.distribution_function.cdf(self.upper_limit, **self.function_inputs)\
                  - self.distribution_function.cdf(self.lower_limit, **self.function_inputs)

            #preparing sampling with the cdf (for truncation)
            yr = np.random.rand(samplesize)*(nrm) \
                 + self.distribution_function.cdf(self.lower_limit, **self.function_inputs)
            xr = self.distribution_function.ppf(yr, **self.function_inputs)
        # if no limits are given, simply return from rvs
        else: 
            xr = self.distribution_function.rvs(**self.function_inputs, size = samplesize)

        return xr
    
    def ppf(self, x:list or float):
        '''ppf (percent point function) values for a given probability.'''
        if hasattr(x, '__iter__'):
            if any(x) < 0 or any(x)>1:
                raise ValueError(f'ppf inputs must be in between 0 and 1.')
        else: 
            if x < 0 or x > 1:
                raise ValueError(f'ppf inputs must be in between 0 and 1.')
        return self.distribution_function.ppf(x, **self.function_inputs)

    def get_pdf(self, x_range:list[float] = None, n_x:int = 10**5, prec:int = 4)->list[np.ndarray]:
        '''
        Creates an x and y array for the pdf of the distribution. 
        If no interval is specified, the first continuous interval of 
        non-zero (rounded to 'prec') values of the pdf is chosen.

        Arguments: 
        - x_range: list of starting and end point for the pdf
        - n_x: number of points in x and y array
        - prec: precision to which the pdf needs to be zero for limits

        Returns: 
        - x and y array: np.ndarrays representing x and y values of PDF
        '''
        # if no x_range is given, we construct one that starts at pi ~0 
        # and ends alike at a higher value where pi ~0, with values 
        # where pi < 0 have to lie in between.
        if x_range is None:
            x_range = [-0.1, 0.1]
            multiplier = 10
            pdf = [1]
            # extend until we have zero values at the ends and non zero values in between.
            #TODO test the automatic x_range
            while (np.round(pdf[0],prec)) > 0 or (np.round(pdf[-1],prec)) > 0 \
                   or len(np.where(np.round(pdf, prec) > 0)[0]) == 0:
                x_range = [multiplier*lim for lim in x_range]
                x = np.linspace(x_range[0], x_range[1], 10**6)
                pdf = self.distribution_function.pdf(x, **self.function_inputs)
            #cut out zero values at the ends: first at the low, then at the hight values
            pi_x = np.round(pdf, prec) # just for readability and speed
            if 0 in pi_x:
                if np.where(pi_x == 0)[0][0] < np.where(pi_x > 0)[0][0]:
                    lower_non_zero = np.where(pi_x > 0)[0][0]
                    pdf = pdf[lower_non_zero:]
                    x = x[lower_non_zero:]
            if 0 in np.round(pdf, prec):
                upper_zero = np.where(np.round(pdf, prec) == 0)[0][0]
                pdf = pdf[:upper_zero]
                x = x[:upper_zero]
            # rescale length of arrays to n_x
            #TODO: check why this doesn't reliably work!
            x = np.linspace(x[0], x[-1], n_x)
            pdf = self.distribution_function.pdf(x, **self.function_inputs)
        else:
            x = np.linspace(x_range[0], x_range[-1], n_x)
            pdf = self.distribution_function.pdf(x, **self.function_inputs)
        return x, pdf

    def plot_samples(self, samplesize:int = 10*4, no_bins:int = 'auto', 
                     show:bool = True)->plt.figure:
        '''Histogram of some (default 10000) samples.'''
        random_values = self.sampling(samplesize)
        fig = plt.figure(figsize=self.figsize)
        plt.hist(random_values, bins = no_bins, histtype = 'bar', color = 'darkorange', 
                 edgecolor = 'k', alpha = 0.8)
        plt.xlabel(f'Values of parameter {self.name} in {self.unit}.')
        plt.ylabel("Number of samples")
        plt.title(f'Parameter  {self.name}: histogram of {samplesize} random values.')
        if show:
            plt.show()
        else:
            plt.close() 
        return fig 
    
    def plot_distribution(self, x_range:list[float] = None, prec:int = 4, show:bool = True)->plt.figure:
        '''Simple PDF plot, (optionally from start - stop in x_range).'''
        x, pdf = self.get_pdf(x_range = x_range,prec = prec)
        fig = plt.figure(figsize=self.figsize)
        plt.plot(x, pdf, label = 'PDF')
        plt.xlabel(f'Values of parameter {self.name} in {self.unit}')
        plt.ylabel(f'probability')
        plt.title(f'Parameter  {self.name}: PDF')
        if show:
            plt.show()
        else:
            plt.close() 
        return fig 
        

    def _find_sts(self) -> tuple:#[sts._continuous_distns.gen, dict]:
        '''
        Assigns a base function from scipys stats library for the chosen 
        distribution which is used for sampling and the pdf. Also sets 
        the base input for these functions with the provided values from
        parameter 1-4. To implement more distributions, just add them 
        as a new elif option.
        '''
        if self.distribution == 'norm':
            mean = self.parameter1
            dev = self.parameter2
            inputs = {'loc' : mean, 'scale' : dev}
            return sts.norm, inputs

        elif self.distribution == 'truncnorm':
            mean = self.parameter1
            my_low = self.parameter2 #actual lower limit
            my_up = self.parameter3 #actual upper limit
            dev = self.parameter4 #standard dev
            #low and up have to be rescaled for the standard norm:
            if dev > 0:
                low, up = (my_low - mean) / dev, (my_up - mean) / dev 
            else: # need to provide low and up in case std is 0...
                low, up = mean-1, mean+1 

            inputs = {'a' : low, 'b' : up, 'loc' : mean, 'scale' : dev}
            return sts.truncnorm, inputs
        
        elif self.distribution == 'lognorm': 
            shape = self.parameter1
            loc = self.parameter2
            scale = self.parameter3
            # shape for how much log it is, loc for the start, scale for the width

            inputs = {'s': shape, 'loc' : loc, 'scale' : scale}
            return sts.lognorm, inputs
        
        #TODO: integrate in tests.
        elif self.distribution == 'randint':
            low = self.parameter1
            high = self.parameter2
            #uniform distribution for all integers between low and high

            inputs = {'low' : low, 'high' : high,}
            return sts.randint, inputs

        elif self.distribution == 'uniform':
            start = self.parameter1, 
            length = self.parameter2
            #uniform distribution from loc to loc+scale

            inputs = {'loc' : start, 'scale' : length}
            return sts.uniform, inputs

        elif self.distribution == 'triang':
            midmultiplier = self.parameter1
            start = self.parameter2
            length = self.parameter3
            #slopes up from loc to loc+shape*scale and down until loc+scale

            inputs = {'c' : midmultiplier, 'loc' : start, 'scale' : length}
            return sts.triang, inputs
        elif self.distribution == 'right skewed':
            '''
            This distribution is a lognormal distribution mirrored at 
            the y-axis. The parameters provided for this 
            '''
            #TODO: test this!
            shape = self.parameter1
            loc = self.parameter2
            scale = self.parameter3

            right_skewed_lognorm = RightSkewedLognorm()
            inputs = {'s': shape, 'loc' : loc, 'scale' : scale}
            return right_skewed_lognorm, inputs
        else:
            raise KeyError(f'The specified distribution ({self.distribution}) seems to '\
                           f'not match any of the implemented distributions.')


        
    def _input_check(self) -> None:
        '''warns (prior to sampling) if values are missing for a distribution'''
        if not self.distribution in self.conditions_dict.keys(): 
            raise KeyError(f'The specified distribution {self.distribution} is not '\
                           f'matching any of the confirmed implemented distributions.')
        else:
            rules = self.conditions_dict[self.distribution]
            for i, (rule, value) in enumerate(zip(rules, [self.parameter1, self.parameter2, 
                                           self.parameter3, self.parameter4])):
                if rule and value is None:
                    raise ValueError(f'For a {self.distribution} distribution a {i}st/nd/rd/th parameter is required.')

    def __repr__(self):
        return f'ScalarParameter named {self.name} (short {self.short_name}, in {self.unit}) with '\
               f'a {self.distribution} distribution with input values {self.function_inputs}.'

class ConstantParameter(UncertainEntity):
    '''
    Parameter class to integrate constant parameters in the MC workflow
    without changing the model itself. This parameter can take any value
    (bool, string, char, lists etc.) and can therefore be used for 
    keyword arguments that cannot be covered else.
    This class can also serve as a blueprint for other custom parameters
    - for example a boolean which is supposed to be True in half of all
    cases and False for the other half.
    '''        

    def __init__(self, name:str, short_name:str, explanation:str, value) -> None:
        self.name = name
        self.short_name = short_name
        self.explanation = explanation
        self.value = value
    
    def ppf(self, x):
        '''Equivalent to ppf function'''
        return [self.value for _ in range(len(x))]

    def sampling(self, samplesize:int)->np.ndarray:
        '''Returns array with copies of original value for the Sampler'''
        return np.array([self.value for _ in range(samplesize)])
    
    def __repr__(self):
        return f'ConstantParameter named {self.name} (short {self.short_name})'\
               f'; explanation: {self.explanation}; value {self.value}.'
