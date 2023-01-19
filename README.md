## Deployment

* Set up secrets:
```
source .env
tanzu secret registry add regsecret --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes  
tanzu secret registry add tap-registry --username ${DATA_E2E_REGISTRY_USERNAME} --password ${DATA_E2E_REGISTRY_PASSWORD} --server https://index.docker.io/v1/ --export-to-all-namespaces --yes
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "registry-credentials"},{"name": "tap-registry"}],"secrets":[{"name": "tap-registry"}]}'
kubectl apply -f config/rbac.yaml
```

* Update the default ClusterStore to include Python and Procfile buildpacks:
```
docker pull registry.tanzu.vmware.com/tanzu-python-buildpack/python:2.3.2 (may need to accept EULA & login to registry.tanzu.vmware.com)
docker tag registry.tanzu.vmware.com/tanzu-python-buildpack/python:2.3.2 oawofolu/tanzu-python-buildpack-full-python:2.3.2
docker push oawofolu/tanzu-python-buildpack-full-python:2.3.2
kp clusterstore add default -b oawofolu/tanzu-python-buildpack-full-python:2.3.2

docker pull registry.tanzu.vmware.com/tanzu-procfile-buildpack/procfile:5.5.0
docker tag registry.tanzu.vmware.com/tanzu-procfile-buildpack/procfile:5.5.0 oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0
docker push oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0
kp clusterstore add default -b oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0

kp clusterstore status default
```

* Deploy the app:
```
tanzu apps workload create image-processor -f config/workload.yaml --yes
```

* Tail the logs:
```
tanzu apps workload tail image-processor --since 64h
```