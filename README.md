## Before you begin:
* Set up a **pre-commit** Git hook which will take care of autogenerating OpenAPI docs:
```
tee -a .git/hooks/pre-commit <<FILE
echo "Generate OpenAPI schema docs................"
$(which python3) -c "from app.analytics import api; api.generate_schema()"
echo "OpenAPI schema docs generated."
echo "Adding OpenAPI schema to repo..."
git add app/analytics/static/api-docs/openapi.json
FILE
```

## Deployment

* Set up secrets:
```
source .env
tanzu secret registry add regsecret --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes  
tanzu secret registry add tap-registry --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "registry-credentials"},{"name": "tap-registry"}],"secrets":[{"name": "tap-registry"}]}'
```

* Include the necessary buildpack dependencies:
```
export TBS_VERSION=1.9.0 # based on $(tanzu package available list buildservice.tanzu.vmware.com --namespace tap-install)
imgpkg copy -b registry.tanzu.vmware.com/tanzu-application-platform/full-tbs-deps-package-repo:${TBS_VERSION} \
--to-repo index.docker.io/oawofolu/tbs-full-deps
tanzu package repository add tbs-full-deps-repository   --url oawofolu/tbs-full-deps:${TBS_VERSION}   --namespace tap-install
tanzu package installed delete full-tbs-deps -n tap-install
tanzu package install full-tbs-deps -p full-tbs-deps.tanzu.vmware.com -v ${TBS_VERSION}  -n tap-install
tanzu package installed get full-tbs-deps   -n tap-install
envsubst < ../tap/resources/tap-values-tbsfull.in.yaml > ../tap/resources/tap-values-tbsfull.yaml
tanzu package installed update tap --values-file ../tap/resources/tap-values-tbsfull.yaml -n tap-install
```

### Deploy the Analytics App

* Deploy the app:
```
source .env
envsubst < config/workload.in.yaml > config/workload.yaml
envsubst < config/workload-api.in.yaml > config/workload-api.yaml
tanzu apps workload create image-processor -f config/workload.yaml --yes
tanzu apps workload create image-processor-api -f config/workload-api.yaml --yes
```

* Tail the logs of the main app:
```
tanzu apps workload tail image-processor --since 64h
```

* Tail the logs of the API app:
```
tanzu apps workload tail image-processor-api --since 64h
```

* Once deployment succeeds, get the URL for the main app:
```
tanzu apps workload get image-processor #should yield image-processor.default.<your-domain>
```

* Get the URL for the API app:
```
tanzu apps workload get image-processor-api #should yield image-processor.default.<your-domain>
```

* To delete the app:
```
tanzu apps workload delete image-processor --yes
tanzu apps workload delete image-processor-api --yes
```

### Deploy the Training Pipeline
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

* View the pipeline in the browser by navigating to http://kubeflow-pipelines.<your-domain-name>

* To delete the pipeline:
```
kapp delete -a image-procesor-pipeline-<THE PIPELINE ENVIRONMENT> -y -nargo
```
