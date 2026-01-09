# -*- coding: utf-8 -*-
import os
import cv2

class BancoImagens:
    def __init__(self, b_n, r_p='./'):
        self.r_p = r_p
        self.b_p = os.path.join(self.r_p, b_n)
        self.a_f = os.path.join(self.b_p, b_n + ".arff")
        self.classes = self._l_s()

    def _l_s(self):
        # Filtro de diretórios operacionais
        i_g = ['results', 'test', 'data', '__pycache__', '.ipynb_checkpoints']
        if not os.path.exists(self.b_p):
            return []
        
        return [d for d in os.listdir(self.b_p)
                if os.path.isdir(os.path.join(self.b_p, d)) 
                and d not in i_g]

    def imagens_da_classe(self, c_n):
        p_c = os.path.join(self.b_p, c_n)
        res = []
        # Formatos suportados pelo motor de visão
        v_e = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

        if not os.path.exists(p_c):
            return []

        for f in os.listdir(p_c):
            if f.lower().endswith(v_e):
                p_f = os.path.join(p_c, f)
                try:
                    m = cv2.imread(p_f)
                    if m is not None and m.size > 0:
                        res.append(m)
                except:
                    continue
        return res