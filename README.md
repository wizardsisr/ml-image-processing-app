## Deployment

* Deploy the app:
```
tanzu apps workload create image-processor -f config/workload.yaml \
  --local-path . \
  --source-image oawofolu/ml-image-processor-source \
  --type web \
  --yes
```

* Tail the logs:
```
tanzu apps workload tail image-processor --since 64h
```