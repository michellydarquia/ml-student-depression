"""
Ponto de entrada único do projeto.

Uso:
    python main.py            # roda o pipeline de ML e depois abre o Streamlit
    python main.py --skip-ml  # pula o pipeline (modelos já treinados) e abre só o Streamlit
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Pula o treinamento (usa os .pkl já existentes em model_artifacts/)",
    )
    args = parser.parse_args()

    # ETAPA 1: Pipeline de Machine Learning
    if not args.skip_ml:
        print("=" * 60)
        print("  ETAPA 1/2 — Pipeline de Machine Learning")
        print("=" * 60)

        from run_pipeline import run_pipeline
        run_pipeline()

        print("\n" + "=" * 60)
        print("  Pipeline finalizado. Iniciando o Streamlit...")
        print("=" * 60 + "\n")
    else:
        print("--skip-ml ativado: pulando treinamento.\n")

        # Verifica se os bundles existem antes de tentar abrir o Streamlit
        for bundle in ["model_artifacts/bundle_sem_leak.pkl",
                       "model_artifacts/bundle_com_leak.pkl"]:
            if not Path(bundle).exists():
                print(f"[ERRO] Arquivo não encontrado: {bundle}")
                print("       Rode sem --skip-ml para treinar os modelos primeiro.")
                sys.exit(1)

    # ETAPA 2: Streamlit
    print("Abrindo o Streamlit em http://localhost:8501")
    print("(pressione Ctrl+C para encerrar)\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "streamlit/app.py"],
        check=True,
    )

if __name__ == "__main__":
    main()