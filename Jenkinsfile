pipeline {
  agent any

  environment {
    DEPLOY_DIR = "/opt/vku-mlops"
    CURRENT_DIR = "/opt/vku-mlops/current"
    VENV_DIR = "/opt/vku-mlops/venv"
    SERVICE = "vku-mlops"
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Setup Deploy Folders') {
      steps {
        sh '''
          set -eux
          sudo mkdir -p ${DEPLOY_DIR}
          sudo mkdir -p ${CURRENT_DIR}
          sudo chown -R jenkins:jenkins ${DEPLOY_DIR}
        '''
      }
    }

    stage('Create/Update Runtime venv') {
      steps {
        sh '''
          set -eux
          python3 -m venv ${VENV_DIR} || true
          ${VENV_DIR}/bin/pip install --upgrade pip
          ${VENV_DIR}/bin/pip install -r requirements.txt
        '''
      }
    }

    stage('Train Model (artifact)') {
      steps {
        sh '''
          set -eux
          ${VENV_DIR}/bin/python src/train.py --data data/data.csv --outdir artifacts --version ${BUILD_NUMBER}
          ls -la artifacts
        '''
      }
    }

    stage('Smoke Test (load model + predict)') {
      steps {
        sh '''
          set -eux
          ${VENV_DIR}/bin/python -c "from src.model import load_model,predict_proba; m,_=load_model('artifacts/model_latest.joblib'); print('prob=',predict_proba(m,[8,7.5,7,6.5,1]))"
        '''
      }
    }

    stage('Deploy to /opt') {
      steps {
        sh '''
          set -eux
          rsync -a --delete ./ ${CURRENT_DIR}/
          # ensure artifacts exist in deployed dir
          test -f ${CURRENT_DIR}/artifacts/model_latest.joblib
        '''
      }
    }

    stage('Install/Update systemd service') {
      steps {
        sh '''
          set -eux
          sudo cp deploy/vku-mlops.service /etc/systemd/system/vku-mlops.service
          sudo systemctl daemon-reload
          sudo systemctl enable vku-mlops.service
          sudo systemctl restart vku-mlops.service
        '''
      }
    }

    stage('Healthcheck') {
      steps {
        sh '''
          set -eux
          sleep 2
          curl -fsS http://localhost:8000/health
          curl -fsS http://localhost:8000/model-info
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'artifacts/*', fingerprint: true
    }
  }
}
