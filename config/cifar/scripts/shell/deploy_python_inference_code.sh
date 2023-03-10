kubectl exec -it ${HOST} -n ${NAMESPACE} --  sh -c "rm -rf ${SHARED_PATH}/mlapp  ${SHARED_PATH}/mlappbase; \
                    git clone ${MLPIPELINE_GIT_REPO} ${SHARED_PATH}/mlapp; \
                    mv ${SHARED_PATH}/mlapp/app ${SHARED_PATH}/mlapp/base_app; \

                    curl -o vendor.tar.gz ${PYFUNC_VENDOR_URI}; \
                    mkdir -p ${SHARED_PATH}/mlapp/_vendor; \
                    tar -xvzf vendor.tar.gz -C ${SHARED_PATH}/mlapp/_vendor --strip-components=1;"