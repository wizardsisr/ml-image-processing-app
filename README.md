## Before you begin:
* Ensure that all pre-requisites described in the **main** branch are satisfied (see README in **main** branch).

### Deploy the Analytics App

* Deploy the app:
```
source .env
envsubst < config/workload.in.yaml > config/workload.yaml
tanzu apps workload create image-processor -f config/workload.yaml --yes
```

* Tail the logs of the main app:
```
tanzu apps workload tail image-processor --since 64h
```

* Once deployment succeeds, get the URL for the main app:
```
tanzu apps workload get image-processor #should yield image-processor.default.<your-domain>
```

* To delete the app:
```
tanzu apps workload delete image-processor --yes
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
