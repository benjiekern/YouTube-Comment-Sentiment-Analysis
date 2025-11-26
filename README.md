 # Sentiment Analysis API on AWS ECS Fargate


This is a project I created to familiarize myself with MLOps. It is a highly available, low-latency API for real-time sentiment analysis, containerized with Docker and deployed to a secure, scalable production environment on AWS ECS Fargate.


# Live Demo

The model is deployed to an internet-facing Application Load Balancer (ALB) and is ready for real-time inference requests. Note: For initial testing, please ensure your request is sent over HTTP (Port 80). Health Check: 

## ```http://sentiment-api-alb-1126856128.us-west-1.elb.amazonaws.com```

For predictions, simply use the following: ```http://sentiment-api-alb-1126856128.us-west-1.elb.amazonaws.com/predict```

Include your text in the following format: 

**{"text": "your_text/comment_goes_here"}**

**Example: {"text": "I really like this video"}**

Then, the model will return the original text, the text after it has been cleaned, and finally its prediction (0 corresponds with a negative sentiment, 1 corresponds with a positive sentiment). 

## To test this on windows command prompt, simply run the following command:


```
curl -X POST ^
  http://sentiment-api-alb-1126856128.us-west-1.elb.amazonaws.com/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"I really like this video\"}"
```

# Engineering Accomplishments

* **ML Model Deployment & Containerization:**
Built a FastAPI inference service (PyTorch) and containerized it with Docker. Deployed to AWS ECS Fargate for real-time production inference.

* **Networking & Connectivity:**
Configured an ALB-backed production endpoint. Resolved external timeout issues by fixing VPC Security Group rules between the ALB and Fargate tasks (Port 80 and 8000).

* **Resource Optimization (Exit Code 137):**
Diagnosed OOM failures in ECS (Exit Code 137) and stabilized the service by allocating proper task memory, preventing restarts.

* **Deployment Reliability:**
Improved release consistency by replacing the :latest image tag with versioned tags across environments.

* **System Stability & Cost Control:**
Stopped an unintended deployment loop that triggered every 4–5 minutes, restoring stable uptime and reducing unnecessary compute/ECR costs.

* **Observability:**
Set up centralized logging using the awslogs driver to stream container output to CloudWatch Logs.

# Tech Stack:

| Category | Technology |
| :--- | :--- |
| **Cloud Platform** | AWS ECS Fargate |
| **Networking & Load Balancing** | AWS Application Load Balancer (ALB) |
| **API Framework** | FastAPI |
| **ML Framework** | PyTorch |
| **Containerization** | Docker |
| **Language** | Python |
| **Observability** | AWS CloudWatch |

#### This project shares similarities with another [project](https://github.com/benjiekern/Neural-Network-for-Review-Sentiment) I worked on. 
