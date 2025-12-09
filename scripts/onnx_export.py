import os
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from modelos import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    create_resnet50,
)


def get_network(params):
    # Model selection based on arguments
    if params.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=3)
    elif params.network == "mobilenetv1":
        model = MobileNetV1(embedding_dim=512)
    elif params.network == "mobilenetv2":
        model = MobileNetV2(embedding_dim=512)
    elif params.network == "mobilenetv3_small":
        model = mobilenet_v3_small(embedding_dim=512)
    elif params.network == "mobilenetv3_large":
        model = mobilenet_v3_large(embedding_dim=512)
    elif params.network == "resnet50":
        model = create_resnet50(embedding_dim=512, pretrained=False)
    else:
        raise ValueError("Unsupported network!")

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument(
        '-w', '--weights',
        default='./weights/mobilenetv2_mcp.pth',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size and input dimensions for ONNX export'
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[LOG] Dispositivo selecionado:", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Caminho dos pesos recebido: {params.weights}")
    print(f"[LOG] Arquitetura selecionada: {params.network}")

    # Inicializa o modelo
    try:
        model = get_network(params)
        print("[LOG] Modelo instanciado com sucesso.")
    except Exception as e:
        print(f"[ERRO] Falha ao instanciar o modelo: {e}")
        return
    model.to(device)

    # Carrega os pesos
    try:
        state_dict = torch.load(params.weights, map_location=device)
        print(f"[LOG] Tipo do checkpoint: {type(state_dict)}. Chaves: {list(state_dict.keys()) if hasattr(state_dict, 'keys') else 'N/A'}")
        try:
            model.load_state_dict(state_dict)
            print("[LOG] Pesos carregados diretamente.")
        except Exception as e:
            if isinstance(state_dict, dict) and 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
                print("[LOG] Pesos carregados da chave 'model'.")
            else:
                print(f"[ERRO] Falha ao carregar os pesos: {e}")
                return
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o checkpoint: {e}")
        return

    # Coloca o modelo em modo de avaliação
    model.eval()

    # Gera o nome do arquivo de saída
    fname = os.path.splitext(os.path.basename(params.weights))[0]
    onnx_model = f'{fname}.onnx'
    print(f"[LOG] Arquivo ONNX será salvo como: {onnx_model}")

    # Cria tensor de entrada dummy
    try:
        x = torch.randn(1, 3, 112, 112).to(device)
        print(f"[LOG] Tensor de entrada dummy criado: {x.shape}")
    except Exception as e:
        print(f"[ERRO] Falha ao criar tensor de entrada: {e}")
        return

    # Prepara dynamic_axes se necessário
    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
        print("[LOG] Exportando modelo com batch dinâmico.")
    else:
        print("[LOG] Exportando modelo com batch fixo.")

    # Exporta para ONNX
    try:
        torch.onnx.export(
            model,
            x,
            onnx_model,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print(f"[SUCESSO] Modelo exportado para {onnx_model}")
    except Exception as e:
        print(f"[ERRO] Falha ao exportar para ONNX: {e}")
        return


if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)
