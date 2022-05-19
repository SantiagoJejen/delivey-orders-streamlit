enter GCP open terminal 

´´´
git pull oriigin master
´´´


´´´
gcloud builds submit \
  --tag gcr.io/$GOOGLE_CLOUD_PROJECT/delivery-streamlit:0.1
´´´

´´´
gcloud run deploy delivery-streamlit \
  --image gcr.io/$PROJECT_ID/delivery-streamlit:0.1
´´´

## Local 

´´´
docker build -t streamlit .
´´´

´´´
docker run -d -p 80:8080 --name my_streamlit streamlit 
´´´