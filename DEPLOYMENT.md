# Deployment Guide for Newsy

This guide will help you deploy the Newsy application using Render (for the backend) and Streamlit Community Cloud (for the frontend).

## Prerequisites

1. A GitHub account with your Newsy repository pushed to it
2. Accounts on [Render](https://render.com/) and [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Your API keys for SERPAPI and Clarifai

## Deploying the Backend to Render

1. **Sign in to Render**
   - Go to [render.com](https://render.com/) and sign in with your account

2. **Create a New Web Service**
   - Click on "New" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your Newsy project

3. **Configure the Web Service**
   - Name: `newsy-backend` (or your preferred name)
   - Environment: `Python`
   - Region: Choose the region closest to your users
   - Branch: `main` (or your default branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m uvicorn mcp.v1.src.main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   - Scroll down to the "Environment Variables" section
   - Add the following variables:
     - `SERPAPI_API_KEY`: Your SerpAPI key
     - `CLARIFAI_API_KEY`: Your Clarifai API key

5. **Deploy the Service**
   - Click "Create Web Service"
   - Wait for the deployment to complete (this may take a few minutes)
   - Once deployed, Render will provide you with a URL (e.g., `https://newsy-backend.onrender.com`)
   - Test the deployment by visiting `https://your-render-url.onrender.com/health`

## Deploying the Frontend to Streamlit Community Cloud

1. **Sign in to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your account

2. **Deploy a New App**
   - Click on "New app"
   - Connect your GitHub repository if you haven't already
   - Select the repository containing your Newsy project
   - Set the main file path to `app.py`
   - Set the Python version (e.g., 3.9)

3. **Set Environment Variables**
   - Click on "Advanced settings"
   - Add the following environment variables:
     - `API_URL`: The URL of your Render backend (e.g., `https://newsy-backend.onrender.com`)
     - `SERPAPI_API_KEY`: Your SerpAPI key
     - `CLARIFAI_API_KEY`: Your Clarifai API key

4. **Deploy the App**
   - Click "Deploy"
   - Wait for the deployment to complete
   - Once deployed, Streamlit will provide you with a URL to access your app

## Verifying the Deployment

1. **Test the Backend API**
   - Visit `https://your-render-url.onrender.com/health` to verify that all services are healthy

2. **Test the Frontend App**
   - Visit your Streamlit app URL
   - Try searching for news articles to ensure the frontend can communicate with the backend

## Troubleshooting

If you encounter any issues:

1. **Backend Issues**
   - Check the Render logs for any error messages
   - Verify that all environment variables are set correctly
   - Ensure that the `/health` endpoint returns a successful response

2. **Frontend Issues**
   - Check the Streamlit logs for any error messages
   - Verify that the `API_URL` environment variable is set correctly
   - Try clearing your browser cache and refreshing the page

3. **Connection Issues**
   - Ensure that the backend is running and accessible
   - Check for any CORS issues (the backend should allow requests from the Streamlit domain)
   - Verify that the API endpoints are correctly formatted in the frontend code
