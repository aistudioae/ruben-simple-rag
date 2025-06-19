# Deploying to Render

This guide explains how to deploy the RAG applications to Render.

## Prerequisites

1. A [Render](https://render.com) account
2. Your API keys ready:
   - OpenAI API key (required for both versions)
   - For Supabase version: Supabase URL and API key
   - For Pinecone version: Pinecone API key and index name

## Option 1: Deploy with Blueprint (Recommended)

1. Fork or clone this repository to your GitHub account
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Blueprint"
4. Connect your GitHub repository
5. Render will detect the `render.yaml` file automatically
6. Add your environment variables in the Render dashboard
7. Click "Deploy"

## Option 2: Manual Deployment

### Deploy Supabase Version

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `simple-rag-supabase`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run simpleRAG.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Add environment variables:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase anon key
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PYTHONUNBUFFERED`: `1`
6. Click "Create Web Service"

### Deploy Pinecone Version

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `simple-rag-pinecone`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements_pinecone.txt`
   - **Start Command**: `streamlit run pinecone_rag.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Add environment variables:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_INDEX_NAME`: Your Pinecone index name
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PYTHONUNBUFFERED`: `1`
6. Click "Create Web Service"

## Switching Between Versions

The `render.yaml` file includes both configurations. By default, only the Supabase version is active. To deploy the Pinecone version instead:

1. Edit `render.yaml`
2. Comment out the Supabase service configuration
3. Uncomment the Pinecone service configuration
4. Commit and push your changes
5. Render will automatically redeploy

## Environment Variables

**Important**: Never commit API keys to your repository! Always set them in the Render dashboard.

### For Supabase Version:
- `SUPABASE_URL`: Found in your Supabase project settings
- `SUPABASE_KEY`: The "anon" key from your Supabase project
- `OPENAI_API_KEY`: From [OpenAI API Keys](https://platform.openai.com/api-keys)

### For Pinecone Version:
- `PINECONE_API_KEY`: From [Pinecone Console](https://app.pinecone.io)
- `PINECONE_INDEX_NAME`: The name of your Pinecone index
- `OPENAI_API_KEY`: From [OpenAI API Keys](https://platform.openai.com/api-keys)

## Post-Deployment

After deployment:

1. Your app will be available at `https://your-app-name.onrender.com`
2. Initial startup may take 2-3 minutes
3. Free tier services spin down after inactivity (cold starts may be slow)
4. Monitor logs in the Render dashboard for any issues

## Troubleshooting

### App won't start
- Check environment variables are set correctly
- Review logs in Render dashboard
- Ensure all dependencies are in requirements file

### API errors
- Verify API keys are valid
- Check API rate limits and quotas
- For Pinecone: ensure index exists and dimensions match (1536)

### Performance issues
- Consider upgrading from free tier for better performance
- Free tier has cold start delays after inactivity

## Cost Considerations

- **Render Free Tier**: 750 hours/month, apps spin down after inactivity
- **OpenAI**: Pay per token usage
- **Supabase**: Free tier includes 500MB database, 2GB bandwidth
- **Pinecone**: Free tier includes 100K vectors

## Security Notes

1. Always use environment variables for sensitive data
2. Enable Render's DDoS protection if available
3. Consider adding authentication for production use
4. Monitor usage to prevent abuse 