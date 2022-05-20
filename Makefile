restart_api_dev:
	docker stop delivery-front  
	docker rm delivery-front  
	docker build -t delivery-streamlit . 
	docker run -d -p 80:8080 --name delivery-front delivery-streamlit

start_api_dev:
	docker run -d -p 80:8080 --name delivery-front delivery-streamlit

stop_api_dev:
	docker stop delivery-front 

init_api_dev:	
	docker build -t delivery-streamlit .
	docker run -d -p 80:8080 --name delivery-front delivery-streamlit

