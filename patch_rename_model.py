for py_file in ["rusket/fpgrowth.py", "rusket/eclat.py", "rusket/mine.py"]:
    with open(py_file) as f:
        content = f.read()

    content = content.replace('class Miner:', 'class Model(ABC):')
    content = content.replace('-> "Miner":', '-> "Model":')
    content = content.replace('class FPGrowth(Miner):', 'class FPGrowth(Model):')
    content = content.replace('class Eclat(Miner):', 'class Eclat(Model):')
    content = content.replace('class AutoMiner(Miner):', 'class AutoMiner(Model):')
    content = content.replace('from .miner import Miner', 'from .model import Model')

    with open(py_file, 'w') as f:
        f.write(content)
