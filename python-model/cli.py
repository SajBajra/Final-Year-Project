import argparse
import os
import sys
import torch


def train(data_dir: str, enhanced: bool = True, epochs: int | None = None, batch_size: int | None = None):
    images = os.path.join(data_dir, 'images')
    labels = os.path.join(data_dir, 'labels.txt')
    if not os.path.exists(images) or not os.path.exists(labels):
        print(f"❌ Dataset not found in {data_dir}. Expect images/ and labels.txt (filename|label)")
        sys.exit(1)
    if enhanced:
        # Run enhanced trainer with args
        cmd = f"{sys.executable} train_crnn_enhanced.py --images {images} --labels {labels}"
        if epochs is not None:
            cmd += f" --epochs {epochs}"
        if batch_size is not None:
            cmd += f" --batch_size {batch_size}"
        print(cmd)
        os.system(cmd)
    else:
        print("Use enhanced trainer; legacy trainer has been deprecated.")
        sys.exit(2)


def infer(model_path: str, chars_file: str, input_path: str, output_file: str | None = None):
    import inference_simple as infer_mod
    if os.path.isfile(input_path):
        pred = infer_mod.predict_image(model_path, chars_file, input_path)
        print(pred)
    elif os.path.isdir(input_path):
        res = infer_mod.predict_folder(model_path, chars_file, input_path, output_file)
        print(f"Processed {len(res)} images")
    else:
        print(f"❌ Input path not found: {input_path}")
        sys.exit(1)


def web(host: str = '0.0.0.0', port: int = 5000):
    import app as webapp
    webapp.app.run(debug=True, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description='Nepali OCR CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train the OCR model')
    p_train.add_argument('--data', default='dataset', help='Dataset directory containing images/ and labels.txt')
    p_train.add_argument('--epochs', type=int, default=None, help='Override number of epochs (optional)')
    p_train.add_argument('--batch-size', type=int, default=None, help='Override batch size (optional)')

    p_infer = sub.add_parser('infer', help='Run inference on image or folder')
    p_infer.add_argument('--model', default='enhanced_crnn_model.pth', help='Model checkpoint path')
    p_infer.add_argument('--chars', default='enhanced_chars.txt', help='Characters file')
    p_infer.add_argument('--input', required=True, help='Image file or folder')
    p_infer.add_argument('--output', default=None, help='Optional output file for folder inference')

    p_web = sub.add_parser('web', help='Run web app')
    p_web.add_argument('--host', default='0.0.0.0')
    p_web.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args.data, enhanced=True, epochs=args.epochs, batch_size=args.batch_size)
    elif args.cmd == 'infer':
        infer(args.model, args.chars, args.input, args.output)
    elif args.cmd == 'web':
        web(args.host, args.port)


if __name__ == '__main__':
    main()


