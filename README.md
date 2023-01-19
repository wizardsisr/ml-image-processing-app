## Deployment

* Set up secrets:
```
source .env
tanzu secret registry add regsecret --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes  
tanzu secret registry add tap-registry --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "registry-credentials"},{"name": "tap-registry"}],"secrets":[{"name": "tap-registry"}]}'
kubectl apply -f config/rbac.yaml
```

* Deploy the app:
```
tanzu apps workload create image-processor -f config/workload.yaml \
  --local-path . \
  --source-image oawofolu/ml-image-processor-source \
  --yes
```

* Tail the logs:
```
tanzu apps workload tail image-processor --since 64h
```