enter GCP open terminal 

```
git pull oriigin master
```


```
gcloud builds submit \
  --tag gcr.io/$GOOGLE_CLOUD_PROJECT/delivery-streamlit:0.1
```

```
gcloud run deploy delivery-streamlit \
  --image gcr.io/$PROJECT_ID/delivery-streamlit:0.1
```

## Local 

try local 

```
docker build -t streamlit .
```

```
docker run -d -p 80:8080 --name my_streamlit streamlit 
```

https://cloud.google.com/community/tutorials/cicd-cloud-run-github-actions

https://www.cloudskillsboost.google/focuses/1038?locale=es&parent=catalog#:~:text=Cree%20una%20cuenta%20de%20servicio,en%20%2B%20Crear%20cuenta%20de%20servicio