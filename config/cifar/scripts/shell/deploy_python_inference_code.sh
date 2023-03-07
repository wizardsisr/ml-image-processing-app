kubectl exec -it ${HOST} -n ${NAMESPACE} --  sh -c "rm -rf ${SHARED_PATH}/mlapp; \
                   git clone ${GIT_REPO} ${SHARED_PATH}/mlapp; \
                   curl -o vendor.tar.gz ${PYFUNC_VENDOR_URI}; \
                   mkdir -p ${SHARED_PATH}/mlapp/_vendor; \
                   tar -xvzf vendor.tar.gz -C ${SHARED_PATH}/mlapp/_vendor --strip-components=1;"