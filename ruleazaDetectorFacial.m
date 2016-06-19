
function [descriptoriExemplePuternicNegative,detectii, scoruriDetectii, imageIdx] = ruleazaDetectorFacial(parametri)
% 'detectii' = matrice Nx4, unde 
%           N este numarul de detectii  
%           detectii(i,:) = [x_min, y_min, x_max, y_max]
% 'scoruriDetectii' = matrice Nx1. scoruriDetectii(i) este scorul detectiei i
% 'imageIdx' = tablou de celule Nx1. imageIdx{i} este imaginea in care apare detectia i
%               (nu punem intregul path, ci doar numele imaginii: 'albert.jpg')

% Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din parametri.numeDirectorExempleTest
% Directorul cu numele parametri.numeDirectorExempleTest contine imagini ce
% pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
% date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
% Functia 'suprimeazaNonMaximele' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
% Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.

% Functia voastra ar trebui sa calculeze pentru fiecare imagine
% descriptorul HOG asociat. Apoi glisati o fereastra de dimeniune paremtri.dimensiuneFereastra x  paremtri.dimensiuneFereastra (implicit 36x36)
% si folositi clasificatorul liniar (w,b) invatat poentru a obtine un scor. Daca acest scor este deasupra unui prag (threshold) pastrati detectia
% iar apoi porcesati toate detectiile prin suprimarea non maximelor.
% pentru detectarea fetelor de diverse marimi folosit un detector multiscale

imgFiles = dir( fullfile( parametri.numeDirectorExempleTest, '*.jpg' ));
if parametri.antrenareCuExemplePuternicNegative  == 1
    imgFiles = dir( fullfile( parametri.numeDirectorExempleNegative, '*.jpg' ));
end
%initializare variabile de returnat
detectii = zeros(0,4);
scoruriDetectii = zeros(0,1);
imageIdx = cell(0,1);
descriptoriExemplePuternicNegative = zeros(0,((parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2)*parametri.dimensiuneDescriptorCelula);

for i = 1:length(imgFiles)      
    fprintf('Rulam detectorul facial pe imaginea %s\n', imgFiles(i).name)
    
    if parametri.antrenareCuExemplePuternicNegative  == 1
        img = imread(fullfile( parametri.numeDirectorExempleNegative, imgFiles(i).name ));
    else
        img = imread(fullfile( parametri.numeDirectorExempleTest, imgFiles(i).name ));    
    end
    
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end 
    
    %initializare variabile locale
    detectii_local = zeros(0,4);
    scoruriDetectii_local = zeros(0,1);
    imageIdx_local = cell(0,1);
    descriptoriExemplePuternicNegative_local = zeros(0,((parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2)*parametri.dimensiuneDescriptorCelula);
    
    
    sz = parametri.dimensiuneFereastra/min(size(img,1),size(img,2));
    scala = 1.5;
    
    while sz < 1.2
        
       imgResize = imresize(img,sz);
        
        descriptorHOGImagine = vl_hog(single(imgResize),parametri.dimensiuneCelulaHOG);
        l = size(descriptorHOGImagine,1);
        c = size(descriptorHOGImagine,2);
        k = parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG;
    
        for j = 1:l - k + 1 
           for l =1:c - k + 1
             
               d = reshape(descriptorHOGImagine(j:j + k - 1,l:l + k - 1,:),1,((parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2)*parametri.dimensiuneDescriptorCelula);
               scor =  d * parametri.w + parametri.b;
               
               if scor > parametri.threshold
                   y_min = (j-1)*parametri.dimensiuneCelulaHOG + 1;
                   x_min = (l-1)*parametri.dimensiuneCelulaHOG + 1;
                   y_max = parametri.dimensiuneFereastra + (j-1)*parametri.dimensiuneCelulaHOG;
                   x_max = parametri.dimensiuneFereastra + (l-1)*parametri.dimensiuneCelulaHOG;

                   y_min = round(y_min/sz);
                   x_min = round(x_min/sz);
                   y_max = round(y_max/sz);
                   x_max = round(x_max/sz);
                   
                   detectii_local = [detectii_local ; [x_min, y_min, x_max, y_max]];
                   scoruriDetectii_local = [scoruriDetectii_local ; scor];
                   imageIdx_local = [imageIdx_local ; imgFiles(i).name];
                   
                  if parametri.antrenareCuExemplePuternicNegative  == 1
                   descriptoriExemplePuternicNegative_local = [descriptoriExemplePuternicNegative_local; d];
                  end
               end
           end
        end

        sz = sz*scala;
    end
    
    if parametri.antrenareCuExemplePuternicNegative  == 1
        img = imread(fullfile( parametri.numeDirectorExempleNegative, imgFiles(i).name ));
    else
        img = imread(fullfile( parametri.numeDirectorExempleTest, imgFiles(i).name ));    
    end
    
    esteMaxim = eliminaNonMaximele(detectii_local, scoruriDetectii_local, [size(img,1) , size(img,2)]);       
    [~,search] = find(esteMaxim == 0);
    detectii_local(search,:) = [];
    scoruriDetectii_local(search, :) = [];
    imageIdx_local(search, :) = [];
    
    if parametri.antrenareCuExemplePuternicNegative  == 1
        descriptoriExemplePuternicNegative_local(search,:) = [];
    end
    
    detectii = [detectii; detectii_local];
    scoruriDetectii = [scoruriDetectii; scoruriDetectii_local];
    imageIdx = [imageIdx; imageIdx_local];
    descriptoriExemplePuternicNegative = [descriptoriExemplePuternicNegative; descriptoriExemplePuternicNegative_local];
end
end




