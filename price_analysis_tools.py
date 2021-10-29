# Import useful packages 
from operator import index
import pandas as pd 
import numpy as np 
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error


# preprocess data and create a unique dataframe containing all the time series called "historical data"

def historical_data(country_code, common_path='data/') :
    specific_path = common_path+country_code+'/'
    electricity_prices = pd.read_csv(specific_path+"prices_"+country_code+".csv", sep=';')
    imports = pd.read_csv(specific_path+"imports_"+country_code+".csv", sep=';')
    exports = pd.read_csv(specific_path+"exports_"+country_code+".csv", sep=';')
    production = pd.read_csv(specific_path+"production_"+country_code+".csv", sep=';')
    demand = pd.read_csv(specific_path+"demand_"+country_code+".csv", sep=';')

    ETS = pd.read_csv(common_path+"EUETSPrices.csv", sep=';')
    coal_prices = pd.read_csv(common_path+"coal_prices.csv", sep=';')
    gas_prices = pd.read_csv(common_path+"bloc_ttf_prices.csv", sep=";")

    production['MTU'] = pd.to_datetime(production['MTU'], utc=True, dayfirst=True)
    production = production.set_index('MTU')

    gas_prices.MTU= pd.to_datetime(gas_prices.MTU, utc=True, dayfirst=True)
    gas_prices = gas_prices.set_index('MTU')
    gas_prices = gas_prices.resample('H').fillna(method='ffill')
    production['Gas_prices'] = gas_prices

    coal_prices.MTU= pd.to_datetime(coal_prices.MTU, utc=True, dayfirst=True)
    coal_prices = coal_prices.set_index('MTU')
    coal_prices = coal_prices.resample('H').fillna(method='ffill')
    production['Coal_prices'] = coal_prices

    ETS.MTU= pd.to_datetime(ETS.MTU)
    ETS = ETS.set_index('MTU')
    ETS = ETS.resample('H').fillna(method='ffill')
    production['ETS'] = ETS

    electricity_prices.MTU= pd.to_datetime(electricity_prices.MTU, utc=True,  dayfirst=True)
    electricity_prices = electricity_prices.set_index('MTU')
    electricity_prices= electricity_prices.loc[~electricity_prices.index.duplicated(), :]
    production['electricity_prices'] = electricity_prices

    imports.MTU= pd.to_datetime(imports.MTU, utc=True,  dayfirst=True)
    imports = imports.set_index('MTU')
    imports= imports.loc[~imports.index.duplicated(), :]
    production['imports'] = imports

    exports.MTU= pd.to_datetime(exports.MTU, utc=True,  dayfirst=True)
    exports = exports.set_index('MTU')
    exports= exports.loc[~exports.index.duplicated(), :]
    production['exports'] = exports

    demand.MTU= pd.to_datetime(demand.MTU, utc=True,  dayfirst=True)
    demand = demand.set_index('MTU')
    demand= demand.loc[~demand.index.duplicated(), :]
    production['demand'] = demand

    production = production.fillna(method='ffill')

    return production

def coeff_reg_lin(dataset_input, months=[4, 5, 6, 7]):
    dataset_input = dataset_input.fillna(method='ffill')
    dataset_input=dataset_input.dropna()

    bool_mask = []
    for i in dataset_input.index.month:
        bool_mask.append(i in months)

    coeff_day = np.zeros((24,14))
    for h in range(24) :
        x_df = dataset_input.loc[(dataset_input.index.hour==h) & bool_mask]
        x = np.array(x_df.drop(columns=['electricity_prices', 'Coal_prices']))
        std_scale = preprocessing.StandardScaler().fit(x)
        x_scaled = std_scale.transform(x)
        Y = np.array(x_df['electricity_prices'])
        reg = linear_model.LinearRegression().fit(x_scaled, Y)
        coeff_day[h]=reg.coef_

    coeff_dataframe = pd.DataFrame(coeff_day, columns=x_df.drop(columns=['electricity_prices', 'Coal_prices']).columns)
    return coeff_dataframe

def forecast(dataset_input,training_months =[6,7,8],forecast_months = [9] ):
    
    bool_mask_training = []
    for i in dataset_input.index.month:
        bool_mask_training.append(i in training_months)

    bool_mask_forecast = []
    for i in dataset_input.index.month:
        bool_mask_forecast.append(i in forecast_months)

    for h in range(24) :
        x_df = dataset_input.loc[(dataset_input.index.hour==h) & bool_mask_training]
        x = np.array(x_df.drop(columns=['electricity_prices', 'Coal_prices']))
        std_scale = preprocessing.StandardScaler().fit(x)
        x_scaled = std_scale.transform(x)
        Y = np.array(x_df['electricity_prices'])
        reg = linear_model.LinearRegression().fit(x_scaled, Y)


        x_test_df=dataset_input.loc[(dataset_input.index.hour==h) & bool_mask_forecast]
        x_test = np.array(x_test_df.drop(columns=['electricity_prices', 'Coal_prices']))
        x_test_scaled = std_scale.transform(x_test)
        y_test = np.array(x_test_df['electricity_prices'])

        y_pred = reg.predict(x_test_scaled)
        print('mse (sklearn): ', np.sqrt(mean_squared_error(y_test,y_pred)))


def interpretable_coeff_reg_lin(dataset_input, months=[4, 5, 6, 7]):
    dataset_input = dataset_input.fillna(method='ffill')
    dataset_input=dataset_input.dropna()

    bool_mask = []
    for i in dataset_input.index.month:
        bool_mask.append(i in months)

    coeff_day = np.zeros((24,14))
    for h in range(24) :
        x_df = dataset_input.loc[(dataset_input.index.hour==h) & bool_mask]
        x = np.array(x_df.drop(columns=['electricity_prices', 'Coal_prices']))
        std_scale = preprocessing.StandardScaler().fit(x)
        x_scaled = std_scale.transform(x)
        Y = np.array(x_df['electricity_prices'])
        reg = linear_model.LinearRegression().fit(x_scaled, Y)
        std_day = np.array(x_df.drop(columns=['electricity_prices', 'Coal_prices']).astype('float').std())
        coeff_day[h] = [reg.coef_[i]/std_day[i] for i in range(len(std_day))]


    coeff_dataframe = pd.DataFrame(coeff_day, columns=x_df.drop(columns=['electricity_prices', 'Coal_prices']).columns)
    return coeff_dataframe