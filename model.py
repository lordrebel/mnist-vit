import einops
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import time
from torch import optim

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn

    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs)+x

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn=fn
        self.norm=nn.LayerNorm(dim)

    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)



class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,dim)
        )

    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.heads=heads
        self.scale=dim**-0.5
        self.to_qkv=nn.Linear(dim,dim*3,bias=False)
        self.to_out=nn.Linear(dim,dim)

    #multihead attention if mask !=None
    def forward(self,x,mask=None):
        b, n, _, h=*x.shape,self.heads
        qkv=self.to_qkv(x) # b n 3*dim
        #to_qkv 通过 fc 将x dim 扩大3 倍，变成三个矩阵 q，k，v
        q, k, v=einops.rearrange(qkv,'b n (qkv h d) -> qkv b h n d' ,qkv=3,h=h)   #3 b h n dim/h
        #矩阵乘
        dots=torch.einsum('bhid,bhjd->bhij',q,k)*self.scale # b h n n
        if mask is not None:
            mask=F.pad(mask.flatten(1),(1,0),value=True)
            assert mask.shape[-1] == dots.shape[-1]
            mask= mask[:, None, :]  *mask[:,:,None]
            #查！
            dots.masked_fill_(~mask,float('-inf'))
            del mask

        attn=dots.softmax(dim=-1)
        out=torch.einsum('bhij,bhjd->bhid',attn,v) # b h n dim/h
        out=einops.rearrange(out, 'b h n d -> b n (h d)') # b n dim
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))
    def forward(self,x,mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self,*,img_size,patch_size,num_classes,dim,depth,heads,mlp_dim,channels=3):
        super().__init__()
        num_patches=(img_size//patch_size)**2
        patch_dim=channels*patch_size**2
        self.patch_size=patch_size
        self.pos_embeding=nn.Parameter(torch.randn(1,num_patches+1,dim))
        self.patch_to_embedding=nn.Linear(patch_dim,dim)
        self.cls_token=nn.Parameter(torch.randn(1,1,dim))
        self.transformer=Transformer(dim,depth,heads,mlp_dim)
        self.to_cls_token=nn.Identity()
        self.mlp_head=nn.Sequential(nn.Linear(dim,mlp_dim),
                                    nn.GELU(),
                                    nn.Linear(mlp_dim,num_classes)
                                    )
    def forward(self,img,mask=None):
        p= self.patch_size #图片切成小图片的大小
        x= einops.rearrange(img,'b c (h p1) (w p2)  -> b (h w) (p1 p2 c)',p1=p,p2=p) #图片维度变换 n c h w->b h*w/(p*p) p*p*c
        x=self.patch_to_embedding(x) #通过个 fc 将tensor 维度： b h*w/(p*p) patchdim -> b h*w/(p*p) dim
        cls_tokens=self.cls_token.expand(img.shape[0],-1,-1) #扩展成  b 1 dim
        x=torch.cat((cls_tokens,x),dim=1) # b h*w/(p*p)+1 dim
        x+=self.pos_embeding
        x=self.transformer(x,mask)
        x= self.to_cls_token(x[:,0])
        return self.mlp_head(x)


torch.manual_seed(42)

DOWNLOAD_PATH = './data'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

N_EPOCHS = 100

start_time = time.time()
'''
patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。'''
model = ViT(img_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
