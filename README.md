## Deployment

* Set up secrets:
```
source .env
tanzu secret registry add regsecret --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes  
tanzu secret registry add tap-registry --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "registry-credentials"},{"name": "tap-registry"}],"secrets":[{"name": "tap-registry"}]}'
```

* Set up Argo:
```
source .env
kubectl create ns argo
kubectl apply -f config/argo-workflow.yaml -nargo
envsubst < config/argo-workflow-http-proxy.in.yaml > config/argo-workflow-http-proxy.yaml
kubectl apply -f config/argo-workflow-http-proxy.yaml -nargo
kubectl create rolebinding default-admin --clusterrole=admin --serviceaccount=argo:default -n argo
kubectl apply -f config/argo-workflow-rbac.yaml -nargo
```

* Login to Argo - copy the token from here:
```
kubectl -n argo exec $(kubectl get pod -n argo -l 'app=argo-server' -o jsonpath='{.items[0].metadata.name}') -- argo auth token
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
tanzu apps workload create image-processor -f config/workload.yaml --yes
```

* Tail the logs:
```
tanzu apps workload tail image-processor --since 64h
```

* Once deployment succeeds, get the URL for the app:
```
tanzu apps workload get image-processor #should yield image-processor.default.<your-domain>
```

* To delete the app:
```
tanzu apps workload delete image-processor --yes
```

### Deploy the Training Pipeline
* cd to </root/of/branch/directory/with/appropriate/model/stage> 
(Example: the **main** github branch represents the "None" stage, the **staging** github branch represents the "Staging" stage, etc)

* Deploy the pipeline:
```
kapp deploy -a image-procesor-pipeline-<THE MODEL STAGE> -f config/cifar/pipeline_app.yaml --logs -y  -nargo
```

* View the pipeline in the browser by navigating to https://argo-workflows.<your-domain-name>

* To delete the pipeline:
```
kapp delete -a image-procesor-pipeline-<THE MODEL STAGE> -y -nargo
```
