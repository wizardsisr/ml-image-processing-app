ssh -i ${SCP_PEM_PATH} ${USER}@${HOST} -o "StrictHostKeyChecking=no" "rm -rf ${SHARED_PATH}/mlapp ${SHARED_PATH}/mlappbase ${SHARED_PATH}/config; \
                   git clone ${MLPIPELINE_GIT_REPO} ${SHARED_PATH}/mlapp; \
                   mv ${SHARED_PATH}/mlapp/app ${SHARED_PATH}/mlapp/base_app; \

                   curl -o vendor.tar.gz ${PYFUNC_VENDOR_URI}; \
                   mkdir -p ${SHARED_PATH}/mlapp/_vendor; \
                   tar -xvzf vendor.tar.gz -C ${SHARED_PATH}/mlapp/_vendor --strip-components=1; \

                   git clone ${GIT_REPO} ${SHARED_PATH}/mlappbase -b ${GIT_REPO_BRANCH}; \
                   mv ${SHARED_PATH}/mlappbase/config ${SHARED_PATH};
                   sed -i \"s/XYZCHANGESETID/$(date +%s)/g; s/XYZDBSCHEMA/${DB_SCHEMA}/g;\" ${SHARED_PATH}/${DB_SCRIPT}"