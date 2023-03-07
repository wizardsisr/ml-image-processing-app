# Generate TLS
source .env
openssl genrsa -out tls.key 2048
openssl req -new -x509 -nodes -days 730 -key tls.key -out tls.crt -config config/db/openssl.conf
kubectl delete secret tls-ssl-postgres -n ${DATA_E2E_POSTGRES_INFERENCE_CLUSTER_NAMESPACE} || true
kubectl create secret generic tls-ssl-postgres --from-file=tls.key --from-file=tls.crt --from-file=ca.crt=tls.crt --namespace ${DATA_E2E_POSTGRES_INFERENCE_CLUSTER_NAMESPACE}

# deploy Postgres cluster
kubectl wait --for=condition=Ready pod -l app=postgres-operator --timeout=120s
kubectl apply -f config/db/postgres-inference-cluster.yaml --namespace ${DATA_E2E_POSTGRES_INFERENCE_CLUSTER_NAMESPACE}