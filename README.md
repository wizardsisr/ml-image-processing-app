## Deployment

* Set up secrets:
```
source .env
tanzu secret registry add regsecret --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes  
tanzu secret registry add tap-registry --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "registry-credentials"},{"name": "tap-registry"}],"secrets":[{"name": "tap-registry"}]}'
```

### Deploy the Analytics App

* Deploy the app:
```
source .env
envsubst < config/workload.in.yaml > config/workload.yaml
envsubst < config/workload-api.in.yaml > config/workload-api.yaml
tanzu apps workload create image-processor-gp-main -f config/workload.yaml --yes
tanzu apps workload create image-processor-gp-main-api -f config/workload-api.yaml --yes
```

* Tail the logs of the main app:
```
tanzu apps workload tail image-processor-gp-main --since 64h
```

* Tail the logs of the API app:
```
tanzu apps workload tail image-processor-gp-main-api --since 64h
```

* Once deployment succeeds, get the URL for the main app:
```
tanzu apps workload get image-processor-gp-main #should yield image-processor-gp-main.default.<your-domain>
```

* Get the URL for the API app:
```
tanzu apps workload get image-processor-api #should yield image-processor-gp-main-api.default.<your-domain>
```

* To delete the app:
```
tanzu apps workload delete image-processor-gp-main --yes
tanzu apps workload delete image-processor-gp-main-api --yes
```

### Set up the Training DB

* Set up Greenplum on AWS: <a href="https://aws.amazon.com/blogs/apn/vmware-greenplum-on-aws-parallel-postgres-for-enterprise-analytics-at-scale/" target="_blank">link</a>

### Set up the Inference DB
* NOTE: If deploying the Postgres instance on GKE, first follow the pre-requisites for deploying VMware Postgres on Kubernetes: <a href="https://docs.vmware.com/en/VMware-SQL-with-Postgres-for-Kubernetes/1.5/vmware-postgres-k8s/GUID-prepare-gke.html" target="_blank">link</a>
* Next, deploy the Postgres instance:
```
source .env
kubectl apply -f config/db/postgres/postgres-inference-cluster.yaml -n ${DATA_E2E_POSTGRES_INFERENCE_CLUSTER_NAMESPACE}
```

* Export the Postgres Inference DB secret:
```
export PGINFERENCE_DB_SECRET=$(kubectl get secret pginstance-inference-db-secret -n ${DATA_E2E_POSTGRES_INFERENCE_CLUSTER_NAMESPACE} -o jsonpath="{.data.password}" | base64 --decode)
```

### Deploy the Training Pipeline
* Setup pipeline credentials by adding them to Vault (they will be injected into the pipeline via the External Secrets controller):
```
source .env
chmod 600 $DATA_E2E_GREENPLUM_PEM
kubectl cp $DATA_E2E_GREENPLUM_PEM vault/vault-0:/tmp
kubectl exec vault-0 -n vault -- vault kv put secret/greenplum/default/training pem=@/tmp/$DATA_E2E_GREENPLUM_PEM password=$DATA_E2E_GREENPLUM_PASSWORD
kubectl exec vault-0 -n vault -- vault kv put secret/postgres/default/inference password=$PGINFERENCE_DB_SECRET
```

* Create ExternalSecrets which will be synced with the pipeline:
```
kubectl apply -f config/cifar/pipeline_external_secrets.yaml -n argo
```

* cd to </root/of/branch/directory/with/appropriate/model/stage> 
(Example: the **main** github branch represents the "main" environment, the **staging** github branch represents the "staging" environment, etc)

* Deploy the pipeline:
```
kapp deploy -a image-procesor-pipeline-<THE PIPELINE ENVIRONMENT> -f config/cifar/pipeline_app.yaml --logs -y  -nargo
```

* View progress:
```
kubectl get app ml-image-processing-pipeline-<THE PIPELINE ENVIRONMENT> -oyaml  -nargo
```

* View the pipeline in the browser by navigating to http://argo-workflows.<your-domain-name>

* To delete the pipeline:
```
kapp delete -a image-procesor-pipeline-<THE PIPELINE ENVIRONMENT> -y -nargo
```
