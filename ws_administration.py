#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade azureml-sdk')


# In[2]:


import azureml.core
from azureml.core import Workspace

# check core SDK version number
print(f'Azure ML SDK Version: {azureml.core.VERSION}')


# In[7]:


# load workspace configuration from the config.json file in the current folder.
try:
    ws = Workspace.from_config()
    #ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    #ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')


# In[4]:


ws.get_details()


# ### Datastores & datasets

# In[15]:


ws.datastores


# In[75]:


# default datastore
from azureml.core import Datastore

Datastore.get_default(ws)


# In[ ]:


# create data store (ADLS)
# https://docs.microsoft.com/fr-fr/python/api/azureml-core/azureml.core.datastore.datastore?view=azure-ml-py#register-azure-data-lake-gen2-workspace--datastore-name--filesystem--account-name--tenant-id-none--client-id-none--client-secret-none--resource-url-none--authority-url-none--protocol-none--endpoint-none--overwrite-false--subscription-id-none--resource-group-none--grant-workspace-access-false-

Datastore.register_azure_data_lake_gen2(workspace, datastore_name, filesystem, account_name, tenant_id=None, client_id=None, client_secret=None, resource_url=None, authority_url=None, protocol=None, endpoint=None, overwrite=False, subscription_id=None, resource_group=None, grant_workspace_access=False)



# In[ ]:


# Azure SQL datastore
# https://docs.microsoft.com/fr-fr/python/api/azureml-core/azureml.data.azure_sql_database_datastore.azuresqldatabasedatastore?view=azure-ml-py

AzureSqlDatabaseDatastore(workspace, name, server_name, database_name, tenant_id=None, client_id=None, client_secret=None, resource_url=None, authority_url=None, username=None, password=None, auth_type=None, service_data_access_auth_identity=None)


# In[ ]:


sql_datastore_name="azuresqldatastore"
   server_name=os.getenv("SQL_SERVERNAME", "<my_server_name>") # Name of the Azure SQL server
   database_name=os.getenv("SQL_DATABASENAME", "<my_database_name>") # Name of the Azure SQL database
   username=os.getenv("SQL_USER_NAME", "<my_sql_user_name>") # The username of the database user.
   password=os.getenv("SQL_USER_PASSWORD", "<my_sql_user_password>") # The password of the database user.

sql_datastore = Datastore.register_azure_sql_database(
   workspace=ws,
   datastore_name=sql_datastore_name,
   server_name=server_name,  # name should not contain fully qualified domain endpoint
   database_name=database_name,
   username=username,
   password=password,
   endpoint='database.windows.net')


# In[16]:


ws.datasets


# In[ ]:


# create tabular dataset
# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

# create a TabularDataset from 3 file paths in datastore
datastore_paths = [(datastore, 'weather/2018/11.csv'),
                   (datastore, 'weather/2018/12.csv'),
                   (datastore, 'weather/2019/*.csv')]

weather_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths) #, set_column_types={'Survived': DataType.to_bool()}


# In[ ]:


# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir='data', target_path='data')


# In[ ]:


# register dataset
titanic_ds = titanic_ds.register(workspace=workspace,
                                 name='titanic_ds',
                                 description='titanic training data')


# In[ ]:


# unregister dataset



# ### Environments

# In[20]:


ws.environments


# In[21]:


ws.images


# In[17]:


ws.linked_services


# In[18]:


ws.private_endpoints


# In[19]:


ws.webservices


# ### List compute targets

# In[9]:


print("Compute instances in your workspace Azure ML:")
cts = ws.compute_targets
for ct in cts:
    print(f'- {ct}')
    
# mélange compute instances et compute clusters => donner une nomenclature


# In[6]:


from azureml.core import Keyvault

#import os
#my_secret = os.environ.get("MY_SECRET")

keyvault = ws.get_default_keyvault()
#keyvault.set_secret(name="sp-authentication-client-secret", value = my_secret)
keyvault.get_secret(name="sp-authentication-client-secret")


# In[6]:


# authenticate with service principal
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id="8e2e7c2d-4702-496d-af6c-96e4bfc9f667", # tenantID
                                    service_principal_id="04beb437-65d4-4853-ac61-8c979f36c29b", # clientId
                                    service_principal_password=keyvault.get_secret(name="sp-authentication-client-secret")) # clientSecret


# In[8]:


from azureml.core import Workspace

ws = Workspace.get(name="sandboxaml",
                   auth=sp,
                   subscription_id="f80606e5-788f-4dc3-a9ea-2eb9a7836082",
                   resource_group="rg-sandbox"
                  )
ws.get_details()


# ### List all pipelines and scheduling

# In[ ]:


from azureml.pipeline.core import Pipeline, PublishedPipeline

published_pipelines = PublishedPipeline.list(ws)

for published_pipeline in published_pipelines:
    print(f"{published_pipeline.name},'{published_pipeline.id}'")

#add scheduling listing


# ### List all the experiments

# In[24]:


from azureml.core import Experiment


# In[66]:


Experiment.get_docs_url()


# In[25]:


Experiment.list(ws, experiment_name=None, view_type='ActiveOnly', tags=None)


# ### Archive, reactivate and delete one experiment

# In[26]:


experiment = Experiment(ws, "Booking_SVD_recsys")


# In[27]:


experiment


# In[39]:


# ajouter un point et appuyer sur Tab pour voir la listes des attributs (instance) et des méthodes (function)
experiment.name


# In[40]:


experiment.archived_time


# In[41]:


experiment.archive()


# In[42]:


experiment.archived_time


# In[43]:


experiment.reactivate()


# In[44]:


experiment.delete(ws, experiment.id)
#Message: Only empty Experiments can be deleted. This experiment contains run(s)


# ### Loop on an archived experiment and delete runs

# In[22]:


from azureml.core import Run


# In[71]:


experiment = Experiment(ws, "Booking_SVD_recsys")

for run in experiment.get_runs():
    print(f'Run {run.id} is on status {run.status}')
    for child in Run(experiment, run.id).get_children():
        print(f'\t child id : {child.id} on status {child.status}')
    
print(f'Nb runs including children : {len(list(experiment.get_runs(include_children=True)))}')


# In[72]:


run_to_del = Run(experiment, '84144200-9587-4d10-ae8a-fcea590e6e92')


# In[ ]:


for exp in Experiment.list(ws, experiment_name=None, view_type='ArchivedOnly', tags=None):
    experiment = Experiment(ws, exp.name)
    for run in experiment.get_runs(include_children=True):
        #'Run' object has no attribute 'delete'
        #run.delete
    experiment.delete(ws, experiment.id)
    #Message: Only empty Experiments can be deleted. This experiment contains run(s)


# ### List experiments with specific tag

# In[20]:


experiment = Experiment(ws, "diabetes_exp")
experiment.set_tags({"dataset":"diabetes"})
experiment.tags


# In[13]:


Experiment.list(ws, experiment_name=None, view_type='ActiveOnly', tags={"dataset":"diabetes"})


# In[14]:


exp_list = Experiment.list(ws, experiment_name=None, view_type='ActiveOnly', tags={"dataset":"diabetes"})
for e in exp_list:
    print(e.name)
    exp = Experiment(ws, e.name)
    exp.tags


# ### Get submittedBy and duration of a run

# In[20]:


runs = experiment.get_runs(include_children=True)
for r in runs:
    print(r.id)


# In[22]:


from azureml.core import Run

run_id = '4bb2ca01-90eb-4fba-8c5c-0304bb55668a'
my_run = Run(experiment, run_id)


# In[23]:


my_run.get_details_with_logs()


# In[24]:


run_logs = my_run.get_details_with_logs()


# In[27]:


run_logs['submittedBy']


# In[28]:


run_logs['startTimeUtc']


# In[36]:


get_ipython().system('pip install arrow')


# In[41]:


import arrow

dt_start = arrow.get(run_logs['startTimeUtc']).datetime
dt_end = arrow.get(run_logs['endTimeUtc']).datetime


# In[42]:


from datetime import datetime

datetime.strptime(run_logs['startTimeUtc'], '%y-%m-%dT%H:%M:%S.%fZ')


# In[45]:


run_length = dt_end - dt_start
run_length


# ## Azure Monitor

# https://docs.microsoft.com/fr-fr/azure/machine-learning/how-to-log-pipelines-application-insights#additional-helpful-queries

# In[47]:


get_ipython().system('pip install opencensus-ext-azure')


# In[48]:


cnx = 'InstrumentationKey=bb414aea-153a-4a4c-a832-f0b0acab4f14;IngestionEndpoint=https://francecentral-0.in.applicationinsights.azure.com/'


# In[50]:


from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Connecting to the workspace and compute target not shown

# Add pip dependency on OpenCensus
dependencies = CondaDependencies()
dependencies.add_pip_package("opencensus-ext-azure>=1.0.1")
run_config = RunConfiguration(conda_dependencies=dependencies)

# Add environment variable with Application Insights Connection String
# Replace the value with your own connection string
run_config.environment.environment_variables = {
    "APPLICATIONINSIGHTS_CONNECTION_STRING": cnx
}


# In[59]:


get_ipython().run_cell_magic('writefile', 'sample_step.py', '\nfrom opencensus.ext.azure.log_exporter import AzureLogHandler\nimport logging\n\nlogger = logging.getLogger(__name__)\nlogger.setLevel(logging.DEBUG)\nlogger.addHandler(logging.StreamHandler())\n\n# Assumes the environment variable APPLICATIONINSIGHTS_CONNECTION_STRING is already set\nlogger.addHandler(AzureLogHandler())\nlogger.warning("I will be sent to Application Insights")\n\n\nfrom azureml.core import Run\n\nrun = Run.get_context(allow_offline=False)\n\ncustom_dimensions = {\n    "parent_run_id": run.parent.id,\n    "step_id": run.id,\n    "step_name": run.name,\n    "experiment_name": run.experiment.name,\n    "run_url": run.parent.get_portal_url(),\n    "run_type": "test"\n}\n\n# Assumes AzureLogHandler was already registered above\nlogger.info("I will be sent to Application Insights with Custom Dimensions", extra= {"custom_dimensions":custom_dimensions})\n')


# In[60]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Use an unique name
cpu_cluster_name = 'paulcluster'

# Tags
clusttags= {"Type": "CPU", 
            "Priority":"Dedicated",
            "Team": "DataScience", 
            "Country": "France"}

try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2',
                                                           vm_priority='dedicated',
                                                           min_nodes = 0, # Min nodes of the cluster
                                                           max_nodes = 2, # Max nodes of the cluster
                                                           tags=clusttags, 
                                                           description="Compute Clusters Std D3V2",
                                                           idle_seconds_before_scaledown=18000) #Timeout for scaling down
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)


# In[61]:


# Configure step with runconfig
sample_step = PythonScriptStep(
        script_name="sample_step.py",
        compute_target=compute_target,
        runconfig=run_config
)


# In[62]:


# Submit new pipeline run
pipeline = Pipeline(workspace=ws, steps=[sample_step])
pipeline.submit(experiment_name="Logging_Experiment")



# In[ ]:




