function descriptoriExempleNegative = obtineDescriptoriExempleNegative(parametri)
% descriptoriExempleNegative = matrice MxD, unde:
%   M = numarul de exemple negative de antrenare (NU sunt fete de oameni),
%   M = parametri.numarExempleNegative
%   D = numarul de dimensiuni al descriptorului
%   in mod implicit D = (parametri.dimensiuneFereastra/parametri.dimensiuneCelula)^2*parametri.dimensiuneDescriptorCelula

imgFiles = dir( fullfile( parametri.numeDirectorExempleNegative , '*.jpg' ));
numarImagini = length(imgFiles);

numarExempleNegative_pe_imagine = round(parametri.numarExempleNegative/numarImagini);
descriptoriExempleNegative = zeros(parametri.numarExempleNegative,(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2*parametri.dimensiuneDescriptorCelula);
disp(['Exista un numar de imagini = ' num2str(numarImagini) ' ce contine numai exemple negative']);
for idx = 1:numarImagini
    disp(['Procesam imaginea numarul ' num2str(idx)]);
    img = imread([parametri.numeDirectorExempleNegative '/' imgFiles(idx).name]);
    if size(img,3) == 3
        img = rgb2gray(img);
    end 
    
    for i=1:numarExempleNegative_pe_imagine
        y = randi([1,size(img,1) - parametri.dimensiuneFereastra]);
        x = randi([1,size(img,2) - parametri.dimensiuneFereastra]);
        descriptorHOGImagine = vl_hog(single(img(y:y+parametri.dimensiuneFereastra - 1,x:x+parametri.dimensiuneFereastra - 1)),parametri.dimensiuneCelulaHOG);
        descriptoriExempleNegative((idx-1)*numarExempleNegative_pe_imagine + i,:) =reshape(descriptorHOGImagine,1,((parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2)*parametri.dimensiuneDescriptorCelula);
    end
end