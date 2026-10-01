"""Create a small ROCm CLI overlay without modifying the pinned upstream source."""
from pathlib import Path
import argparse
import subprocess

REVISION = 'a57c05bfe2a91b5e0cb0983479634eba3e28ede5'


def prepare(source, out):
    source = Path(source)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('SenseVoice source does not match the pinned release')
    path = source / 'runtime/llama.cpp/sensevoice/funasr-sensevoice/funasr-sensevoice.cpp'
    text = path.read_text()
    changes = [
        ('static ggml_backend_dev_t find_gpu_backend_device',
         'static int device_index = 0;\n\nstatic ggml_backend_dev_t find_gpu_backend_device'),
        ('  ggml_backend_dev_t integrated_fallback=nullptr;',
         '  int ordinal=0;'),
        ('if(type==GGML_BACKEND_DEVICE_TYPE_GPU) return dev;',
         'if(ordinal++==device_index) return dev;'),
        ('      if(!integrated_fallback) integrated_fallback=dev;', ''),
        ('  return integrated_fallback;', '  return nullptr;'),
        ('  } else if(name=="cuda"){', '''  } else if(name=="rocm"){
    ggml_backend_dev_t dev=find_gpu_backend_device("rocm");
    if(!dev){
      fprintf(stderr,"ROCm device %d unavailable; build with GGML_HIP=ON\\n",device_index);
      exit(1);
    }
    return initialize_device_backend(name,dev);
  } else if(name=="cuda"){'''),
        ('    else if(!strcmp(argv[i],"--ids"))ids_mode=true;', '''    else if(!strcmp(argv[i],"--device")&&i+1<argc){
      char *end=nullptr;
      long value=strtol(argv[++i],&end,10);
      if(*end || value<0 || value>65535){fprintf(stderr,"invalid device\\n");return 2;}
      device_index=(int)value;
    }
    else if(!strcmp(argv[i],"--ids"))ids_mode=true;'''),
    ]
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError(f'Unsupported SenseVoice source boundary: {old}')
        text = text.replace(old, new)
    text = text.replace('cpu|cuda|vulkan', 'cpu|cuda|rocm|vulkan')
    text = text.replace('[--srt] [--ids]', '[--device N] [--srt] [--ids]')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.out)
