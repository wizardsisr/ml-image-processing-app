# ML Image Processing App

## Deploy the Analytics App

* Deploy the app:
```
source .env
envsubst < config/workload-api.in.yaml > config/workload-api.yaml
tanzu apps workload create image-processor-api -f config/workload-api.yaml --yes
```

* Tail the logs of the API app:
```
tanzu apps workload tail image-processor-api --since 64h
```

* Once deployment succeeds, get the URL for the main app:
```
tanzu apps workload get image-processor-api #should yield image-processor-api.default.<your-domain>
```

* To delete the app:
```
tanzu apps workload delete image-processor-api --yes
```