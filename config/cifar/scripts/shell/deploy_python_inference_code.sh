kubectl exec -it ${HOST} -n ${NAMESPACE} --  sh -c "rm -rf ${SHARED_PATH}/mlapp  ${SHARED_PATH}/mlappbase; \
                   git clone ${GIT_REPO} ${SHARED_PATH}/mlapp; \

                   curl -o vendor.tar.gz ${PYFUNC_VENDOR_URI}; \
                   mkdir -p ${SHARED_PATH}/mlapp/_vendor; \
                   tar -xvzf vendor.tar.gz -C ${SHARED_PATH}/mlapp/_vendor --strip-components=1; \

                   git clone ${GIT_REPO} ${SHARED_PATH}/mlappbase; \
                   mv ${SHARED_PATH}/mlappbase/config ${SHARED_PATH};
                   sed -i \"s/XYZCHANGESETID/$(date +%s)/g; s/XYZDBSCHEMA/${DB_SCHEMA}/g;\" ${SHARED_PATH}/${DB_SCRIPT}"