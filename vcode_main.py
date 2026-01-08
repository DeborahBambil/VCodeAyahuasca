import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tempfile # Para gerenciar arquivos sem sudo

# Importação dos scripts
try:
    from bancoImagens import BancoImagens  
    from extratores import Extratores
    from arff import Arff
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")

# Weka Wrapper
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
from weka.core.classes import Random

class VCodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VCode Ayahuasca - Standard User Mode")
        self.geometry("900x700")
        self.input_dir = ""
        self.jvm_started = False
        
        self.algorithms = {
            "Random Forest": "weka.classifiers.trees.RandomForest",
            "SVM (SMO)": "weka.classifiers.functions.SMO",
            "KNN (IBk)": "weka.classifiers.lazy.IBk",
            "J48": "weka.classifiers.trees.J48"
        }
        
        self.setup_ui()
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                # Adicionado parêmetro para evitar criação de logs em pastas protegidas
                jvm.start(max_heap_size="2048m")
                self.jvm_started = True
                self.log(">>> Sistema Neural (JVM) pronto.")
        except Exception as e: 
            self.log(f"Erro ao carregar JVM: {e}")

    def setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text="VCode Ayahuasca", font=("Arial", 20, "bold"))
        self.lbl_logo.pack(pady=20)

        self.btn_open = ctk.CTkButton(self.sidebar, text="Select Dataset", command=self.select_dir)
        self.btn_open.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="Algorithm:").pack(pady=(20, 0))
        self.algo_menu = ctk.CTkOptionMenu(self.sidebar, values=list(self.algorithms.keys()))
        self.algo_menu.pack(pady=10, padx=20)
        
        self.btn_run = ctk.CTkButton(self.sidebar, text="RUN ANALYSIS", fg_color="green", hover_color="darkgreen", command=self.start)
        self.btn_run.pack(pady=30, padx=20)

        self.console = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.console.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def log(self, msg):
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_dir(self):
        # filedialog.askdirectory geralmente não exige sudo
        path = filedialog.askdirectory()
        if path:
            self.input_dir = path
            self.log(f"Diretório selecionado: {path}")

    def start(self):
        if not self.input_dir:
            messagebox.showwarning("Atenção", "Selecione o diretório primeiro!")
            return
        if not self.jvm_started:
            messagebox.showwarning("Aguarde", "Inicializando Java...")
            return
            
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        try:
            self.log("\n" + "="*40)
            self.log("--- INICIANDO PROCESSAMENTO ---")
            
            parent = os.path.dirname(self.input_dir)
            base = os.path.basename(self.input_dir)
            
            banco = BancoImagens(base, parent) 
            ext = Extratores()
            
            dados, nomes_at, tipos_at = [], [], []
            classes_encontradas = banco.classes
            
            if not classes_encontradas:
                raise Exception("Nenhuma pasta de classe encontrada.")

            for classe in classes_encontradas:
                imgs = banco.imagens_da_classe(classe)
                self.log(f"Processando '{classe}': {len(imgs)} imagens.")
                
                for img in imgs:
                    resultado = ext.extrai_todos(img)
                    if resultado and len(resultado) == 3:
                        n, t, v = resultado
                        if n:
                            if not nomes_at: nomes_at, tipos_at = n, t
                            dados.append(v + [classe])
            
            if not dados:
                raise Exception("Nenhum dado extraído.")
            
            # --- AJUSTE ANTI-SUDO ---
            # Tenta salvar no diretório do usuário caso o diretório do dataset esteja bloqueado
            try:
                arff_path = os.path.join(self.input_dir, f"{base}.arff")
                Arff().cria(arff_path, dados, base, nomes_at, tipos_at, classes_encontradas)
            except PermissionError:
                # Se falhar por falta de sudo, salva na pasta de usuário
                self.log("Aviso: Sem permissão na pasta do dataset. Salvando em documentos...")
                user_home = os.path.expanduser("~")
                arff_path = os.path.join(user_home, f"VCode_Result_{base}.arff")
                Arff().cria(arff_path, dados, base, nomes_at, tipos_at, classes_encontradas)

            self.log(f"Arquivo ARFF pronto em: {arff_path}")
            self.run_weka(arff_path)

        except Exception as e:
            self.log(f"ERRO: {str(e)}")
        finally:
            self.btn_run.configure(state="normal")

    def run_weka(self, path):
        try:
            self.log("\n--- VALIDANDO (10-FOLD CV) ---")
            loader = Loader(classname="weka.core.converters.ArffLoader")
            data = loader.load_file(path)
            data.class_is_last()
            
            algo_selecionado = self.algo_menu.get()
            cls = Classifier(classname=self.algorithms[algo_selecionado])
            evl = Evaluation(data)
            
            evl.crossvalidate_model(cls, data, 10, Random(1))
            
            self.log("\n" + "*"*30)
            self.log(f"ACURÁCIA: {evl.percent_correct:.2f}%")
            self.log(f"MATRIZ DE CONFUSÃO:\n{evl.matrix()}")
            self.log("*"*30)
            
            messagebox.showinfo("VCode", "Análise Concluída com Sucesso!")

        except Exception as e:
            self.log(f"Erro Weka: {e}")

if __name__ == "__main__":
    app = VCodeApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()