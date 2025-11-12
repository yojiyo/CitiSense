#!/bin/bash
# setup.sh - Production deployment script

echo "Setting up CitiSense Dashboard for production..."

# Create necessary directories
mkdir -p data model logs

# Set proper permissions
chmod 755 data model logs

# Create systemd service files for production
sudo tee /etc/systemd/system/citisense-backend.service > /dev/null <<EOF
[Unit]
Description=CitiSense FastAPI Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/citisense
Environment=PYTHONPATH=/opt/citisense
ExecStart=/opt/citisense/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/citisense-frontend.service > /dev/null <<EOF
[Unit]
Description=CitiSense Streamlit Frontend
After=network.target citisense-backend.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/citisense
Environment=PYTHONPATH=/opt/citisense
ExecStart=/opt/citisense/venv/bin/streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Nginx configuration
sudo tee /etc/nginx/sites-available/citisense > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    # Frontend (Streamlit)
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Backend API
    location /api {
        rewrite ^/api(.*) \$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    client_max_body_size 100M;
}
EOF

echo "Configuration files created!"
echo "Remember to:"
echo "1. Update your-domain.com in nginx config"
echo "2. Run: sudo systemctl enable citisense-backend citisense-frontend"
echo "3. Run: sudo systemctl start citisense-backend citisense-frontend"
echo "4. Run: sudo ln -s /etc/nginx/sites-available/citisense /etc/nginx/sites-enabled/"
echo "5. Run: sudo nginx -t && sudo systemctl reload nginx"