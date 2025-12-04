from markdown_pdf import MarkdownPdf, Section

def converter_markdown_para_pdf(caminho_arquivo_md, caminho_arquivo_pdf):
    """
    Converte um arquivo Markdown (.md) para um arquivo PDF (.pdf).
    """
    try:
        # Ler o conteúdo do arquivo Markdown
        with open(caminho_arquivo_md, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Criar um objeto MarkdownPdf
        pdf = MarkdownPdf()
        
        # Adicionar metadados ao PDF
        pdf.meta['title'] = 'Relatório Final - Sistema de Reconhecimento de Plantas'
        pdf.meta['author'] = 'Fabio Batista Rodrigues'

        # Adicionar o conteúdo Markdown como uma seção do PDF
        pdf.add_section(Section(markdown_content, toc=False))

        # Salvar o PDF no caminho especificado
        pdf.save(caminho_arquivo_pdf)

        print(f"✅ Conversão concluída com sucesso!")
        print(f"Arquivo PDF gerado: {caminho_arquivo_pdf}")

    except FileNotFoundError:
        print(f"❌ Erro: O arquivo de entrada não foi encontrado em: {caminho_arquivo_md}")
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a conversão: {e}")

if __name__ == "__main__":
    arquivo_md_entrada = "REPORT.md"
    arquivo_pdf_saida = "REPORT.pdf"
    converter_markdown_para_pdf(arquivo_md_entrada, arquivo_pdf_saida)