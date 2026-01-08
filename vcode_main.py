import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Importação dos scripts
# Certifique-se que os nomes dos arquivos .py são exatamente esses
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
        self.title("VCode - 10-Fold Cross Validation")
        self.geometry("900x700")
        self.input_dir = ""
        self.jvm_started = False
        
        # Mapeamento de Algoritmos
        self.algorithms = {
            "Random Forest": "weka.classifiers.trees.RandomForest",
            "SVM (SMO)": "weka.classifiers.functions.SMO",
            "KNN (IBk)": "weka.classifiers.lazy.IBk",
            "J48": "weka.classifiers.trees.J48"
        }
        
        self.setup_ui()
        # Inicia o núcleo Java
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                jvm.start(max_heap_size="2048m")
                self.jvm_started = True
                self.log(">>> Sistema Neural (JVM) pronto.")
        except Exception as e: 
            self.log(f"Erro ao carregar JVM: {e}")

    def setup_ui(self):
        # Sidebar - Menu Lateral
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text="VCode Ayahuasca", font=("Arial", 20, "bold"))
        self.lbl_logo.pack(pady=20)

        self.btn_open = ctk.CTkButton(self.sidebar, text="Selecionar Dataset", command=self.select_dir)
        self.btn_open.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="Algoritmo:").pack(pady=(20, 0))
        self.algo_menu = ctk.CTkOptionMenu(self.sidebar, values=list(self.algorithms.keys()))
        self.algo_menu.pack(pady=10, padx=20)
        
        self.btn_run = ctk.CTkButton(self.sidebar, text="INICIAR ANÁLISE", fg_color="green", hover_color="darkgreen", command=self.start)
        self.btn_run.pack(pady=30, padx=20)

        # Main Console
        self.console = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.console.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def log(self, msg):
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_dir(self):
        self.input_dir = filedialog.askdirectory()
        if self.input_dir:
            self.log(f"Diretório selecionado: {self.input_dir}")

    def start(self):
        if not self.input_dir:
            messagebox.showwarning("Atenção", "Selecione o diretório primeiro!")
            return
        if not self.jvm_started:
            messagebox.showwarning("Aguarde", "Aguardando inicialização do núcleo Java...")
            return
            
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        try:
            self.log("\n" + "="*40)
            self.log("--- INICIANDO EXTRAÇÃO DE ATRIBUTOS ---")
            
            # Ajuste de caminhos
            parent = os.path.dirname(self.input_dir)
            base = os.path.basename(self.input_dir)
            
            # CORREÇÃO: Nome da Classe Importada (BancoImagens)
            banco = BancoImagens(base, parent) 
            ext = Extratores()
            
            dados, nomes_at, tipos_at = [], [], []
            classes_encontradas = banco.classes
            
            if not classes_encontradas:
                raise Exception("Nenhuma pasta de classe encontrada dentro do diretório.")

            for classe in classes_encontradas:
                imgs = banco.imagens_da_classe(classe)
                self.log(f"Processando '{classe}': {len(imgs)} imagens encontradas.")
                
                for img in imgs:
                    # Tenta extrair os dados da imagem
                    resultado = ext.extrai_todos(img)
                    
                    if resultado is not None and len(resultado) == 3:
                        n, t, v = resultado
                        if n: # Garante que a lista de nomes não está vazia
                            if not nomes_at: 
                                nomes_at, tipos_at = n, t
                            dados.append(v + [classe])
            
            if not dados:
                raise Exception("O processo terminou, mas nenhum dado foi extraído das imagens.")
            
            # Gerar o ARFF
            arff_path = os.path.join(self.input_dir, f"{base}.arff")
            Arff().cria(arff_path, dados, base, nomes_at, tipos_at, classes_encontradas)
            self.log(f"Arquivo ARFF gerado com sucesso: {arff_path}")
            
            # Rodar Classificação
            self.run_weka(arff_path)

        except Exception as e:
            self.log(f"ERRO: {str(e)}")
        finally:
            self.btn_run.configure(state="normal")

    def run_weka(self, path):
        try:
            self.log("\n--- VALIDANDO (10-FOLD CROSS VALIDATION) ---")
            
            # Carregar dados no Weka
            loader = Loader(classname="weka.core.converters.ArffLoader")
            data = loader.load_file(path)
            data.class_is_last()
            
            # Configurar Algoritmo Selecionado
            algo_selecionado = self.algo_menu.get()
            self.log(f"Algoritmo: {algo_selecionado}")
            
            cls = Classifier(classname=self.algorithms[algo_selecionado])
            evl = Evaluation(data)
            
            # Executar Validação de 10 Dobras
            evl.crossvalidate_model(cls, data, 10, Random(1))
            
            # Exibir Resultados
            self.log("\n" + "*"*30)
            self.log(f"ACURÁCIA: {evl.percent_correct:.2f}%")
            self.log(f"F-MEASURE: {evl.weighted_f_measure:.3f}")
            self.log("\nMATRIZ DE CONFUSÃO:")
            self.log(evl.matrix())
            self.log("*"*30)
            
            messagebox.showinfo("VCode", f"Análise Concluída!\nAcurácia: {evl.percent_correct:.2f}%")

        except Exception as e:
            self.log(f"Erro na Classificação: {e}")

if __name__ == "__main__":
    app = VCodeApp()
    # Garante que a JVM feche ao sair
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()