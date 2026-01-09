# -*- coding: utf-8 -*-
import io

class Arff(object):
    def cria(self, f_n, d_s, r_n, a_n, a_t, c_n):
        """
        Gera estrutura de dados formatada.
        """
        with open(f_n, 'w', encoding='utf-8') as f:
            # Cabeçalho da relação
            f.write(f"@relation {r_n}\n\n")

            # Definição dos atributos mapeados
            for n, t in zip(a_n, a_t):
                f.write(f"@attribute {n} {t}\n")

            # Definição da classe alvo
            f.write(f"@attribute target {{{','.join(c_n)}}}\n\n")

            # Seção de dados
            f.write('@data\n\n')

            for r in d_s:
                # Concatenação e escrita da linha de instância
                f.write(",".join(map(str, r)) + "\n")