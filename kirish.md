# Maqsad  Learning/Deep Learning haqida boshlang'ich tushuncha;
# PyTorch kutubxonasini qo'llash qobilyatini shaklantirish.
# Kimlar uchun:
#   * Istalgan inson o'rgana oladi
#        * Elementar Algebra +  Ehtimollar nazaryasi
#        * Boshlang'ich python







# Machine Learning nima?
# Machine Learning - bu mashinaviy o'rganish bu ko'pincha raqamli malumotlarni bashorat qilish uchun ishlatiladi


# Pytorch mavzulari

# PyTorch asoslari
 * PyTorch overview (NIma va qayerda ishlatiladi)
 * Tensor tushunchasi
   * Tensor yaratish usullari (torch.tensor , torch.zeros , torch.ones, torch.rand)
   * Tensor shape,  o'lchamlari (view, reshape)
   * Indexing, slicing, concatenation
   * GPU (CUDA) va CPUda ishlatish
 * Tensor va Numpy bilan ishlash (convert qilish)


# Autograd (Automatic Differentiation)
* requires_grad tushunchasi
* Forward va backward pass
* backward() va grad
* Gradientlarni nolga tushurish (optimizer.zero_grad())
* Conputational graph tushunchasi



# Neural Network Fundamentals
* torch.nn moduli bilan tanishish
* nn.Module va custom model yaratish
* nn.Linear, nn.Conv2d , nn.ReLU, nn.Sigmoid, nn.Softmax
* forward() funksiyasi
* torch.nn.functional (F modulida funksiyalar)



# Dataset va DataLoader
* torch.utils.data.Dataset va Dataloder
* Custom dataset yaratish
* Batch, Shuffle, Sampler tushunchalari
* torchvision.datasets va transformlar  (transforms.Compose)


# Loss Functions
* Regression uchun MSELoss , L1Loss
* Classification uchun: CrossEntropyLoss, NLLLoss, BCELoss
* Kenroq loss funksiyalari (SmoothL1Loss , HuberLoss)


# Optimizer
* Gredient Descent tushunchasi
* PyTorch optimizerlari:
  * SGD , Adam , RMSprop, Adagrad
* Learning Rate tushunchasi
* Learning Rate Schedulers (SpetLR , RediceLROnPlateau)


# Traning Loop
* Forward pass
* Loss hisoblash
* Backward pass
* Gradient update qilish
* Epoch va batch sikllari
* Traning va Validation ajratish

# Modelni saqlash va yuklash
* torch.save va torch.load
* state_dict tushunchasi
* Modelni checkpoint qilish
* Fine-tuning va transfer learning


# GPU bilan ishlash
* CUDA tekshirish (torch.cuda.is_available())
* Tensor va model GPUga o'tkazish (to(device))
* Multiple GPU (DataParallel, DistributedDataParallel)


# Kompyuter ko'rish (Compyuter vision)
* torchvision kutubxonasi
* 
* Datasetlar: CIFAR-10, MNIST
* 
* Convolutional Neural Networks (CNN)
* 
* Pooling (MaxPool, AvgPool)
* 
* Data Augmentation (RandomCrop, RandomFlip)
* 
* Transfer Learning (ResNet, VGG, EfficientNet)

# Natural Language Processing (NLP)

* torchtext bilan ishlash
* Embeddings (nn.Embedding)
* RNN, LSTM, GRU arxitekturalari
* Seq2Seq modellar
* Transformer arxitekturasi (BERT, GPT, T5)

# Advanced Topics

* Custom Loss Functions yozish
* Custom Optimizer yozish
* Gradient Clipping
* Mixed Precision Training (torch.cuda.amp)
* Model quantization va pruning
* ONNX export va boshqa frameworklarda ishlatish

# PyTorch Ecosystem

* torchvision (CV uchun)
* torchtext (NLP uchun)
* torchaudio (audio uchun)
* ignite, lightning (trainingni avtomatlashtirish uchun)
* HuggingFace Transformers bilan PyTorch

# Deployment

* TorchScript (modelni C++ ga eksport qilish)
* ONNX formatiga o‘tkazish
* Mobil uchun PyTorch Mobile
* Model serving (API orqali ishlatish)