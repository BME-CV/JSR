        

        self.cconv_5 = nn.Sequential(
            ConvInsBlock(16*2* c, 16*c, 3, 1),  
        )
        # warp scale 2
        self.defconv5 = nn.Conv3d(16*2* c, 3, 3, 1, 1)
        self.defconv5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv5.weight.shape))
        self.defconv5.bias = nn.Parameter(torch.zeros(self.defconv5.bias.shape))
        
        self.dconv5 = nn.Sequential(
            ConvInsBlock(16*3* c, 16 * c),
        )

        self.upconv4 = UpConvBlock(16*c, 8*c, 4, 2)
      


        self.cconv_4 = nn.Sequential(
            ConvInsBlock(8*3* c, 8*c, 3, 1),
            
        )
    
        self.defconv4 = nn.Conv3d(8* c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
       

        self.upconv3 = UpConvBlock(8* c, 4* c, 4, 2)
        self.cconv_3 = CConv(4* 3 * c,4*c)

     
        self.defconv3 = nn.Conv3d(4*c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        

        self.upconv2 = UpConvBlock(4* c, 2* c, 4, 2)
        self.cconv_2 = CConv(2*3*c,2* c)

       
        self.defconv2 = nn.Conv3d(2*c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        

        self.upconv1 = UpConvBlock(2*c, c, 4, 2)
        self.cconv_1 = CConv(3*c,c)

       
        self.defconv1 = nn.Conv3d(c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))
       