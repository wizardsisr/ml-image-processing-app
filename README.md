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
# docker pull registry.tanzu.vmware.com/tanzu-python-buildpack/python:2.3.2 (may need to accept EULA & login to registry.tanzu.vmware.com)
# docker tag registry.tanzu.vmware.com/tanzu-python-buildpack/python:2.3.2 oawofolu/tanzu-python-buildpack-full-python:2.3.2
# docker push oawofolu/tanzu-python-buildpack-full-python:2.3.2
# kp clusterstore add default -b oawofolu/tanzu-python-buildpack-full-python:2.3.2

# docker pull registry.tanzu.vmware.com/tanzu-procfile-buildpack/procfile:5.5.0
# docker tag registry.tanzu.vmware.com/tanzu-procfile-buildpack/procfile:5.5.0 oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0
# docker push oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0
# kp clusterstore add default -b oawofolu/tanzu-procfile-buildpack-full-procfile:5.5.0

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