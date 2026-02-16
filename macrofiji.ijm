/*
 * COMPARATORE FIJI "SMART MATCHING"
 * Gestisce nomi file leggermente diversi e calcola l'errore.
 */

// --- CONFIGURAZIONE (ATTENTO AGLI SLASH FINALI /) ---
dirGT = "/home/ricc/Documents/PatchesTests/107GT/";  // Dove hai le maschere manuali
dirAI = "/home/ricc/Documents/PatchesTests/107Unett/";             // Dove hai le maschere AI

// Dove salvare il CSV
savePath = "/home/ricc/Documents/PatchesTests/confronto_finale2.csv";

// Fattore di correzione (Bootstrap)
k_factor = 1.38;

// Funzione per misurare (con pulizia 50px)
function misura(path) {
    if (!File.exists(path)) return newArray(0, 0);
    open(path);
    run("8-bit");
    setThreshold(2, 255); // Prende solo il DANNO (valore 2)
    run("Analyze Particles...", "size=0-Infinity display clear"); // PULIZIA QUI!
    
    totArea = 0;
    avgCirc = 0;
    if (nResults > 0) {
        for (j=0; j<nResults; j++) {
            totArea += getResult("Area", j);
            avgCirc += getResult("Circ.", j);
        }
        avgCirc = avgCirc / nResults;
    }
    close();
    return newArray(totArea, avgCirc);
}

// Funzione che cerca il file AI anche se il nome è un po' diverso
function trovaFileAI(nomeGT, cartellaAI) {
    // Caso facile: nomi uguali
    if (File.exists(cartellaAI + nomeGT)) return nomeGT;
    
    // Caso difficile: cerca file che INIZIANO con lo stesso codice (es H508_h022)
    baseName = substring(nomeGT, 0, lengthOf(nomeGT)-4); // Toglie .png
    listaAI = getFileList(cartellaAI);
    
    for (k=0; k<listaAI.length; k++) {
        // Se il file AI inizia con il nome del GT (es. aggiunge -1.png)
        if (startsWith(listaAI[k], baseName)) {
            return listaAI[k];
        }
    }
    return "NON_TROVATO";
}

// --- INIZIO SCRIPT ---
run("Set Measurements...", "area shape redirect=None decimal=3");
f = File.open(savePath);
print(f, "File,Area_GT,Area_AI_Raw,Area_AI_Corr,Errore_%,Circ_GT,Circ_AI");

listGT = getFileList(dirGT);
setBatchMode(true); // Non mostra le immagini a schermo (più veloce)

count = 0;
for (i=0; i<listGT.length; i++) {
    fileGT = listGT[i];
    if (endsWith(fileGT, ".png")) {
        
        // 1. Trova il file AI corrispondente
        fileAI = trovaFileAI(fileGT, dirAI);
        
        if (fileAI != "NON_TROVATO") {
            // 2. Misura entrambi
            datiGT = misura(dirGT + fileGT);
            datiAI = misura(dirAI + fileAI);
            
            aGT = datiGT[0];
            aAI = datiAI[0];
            
            // Se entrambi hanno trovato danno
            if (aGT > 0 && aAI > 0) {
                // 3. Applica correzione
                aAI_corr = aAI * k_factor;
                err = (aAI_corr - aGT) / aGT * 100;
                
                // Scrive nel CSV
                print(f, fileGT + "," + aGT + "," + aAI + "," + aAI_corr + "," + err + "," + datiGT[1] + "," + datiAI[1]);
                count++;
                print("Processato: " + fileGT + " (Match con: " + fileAI + ")");
            }
        } else {
            print("SALTATO: Non trovo corrispondenza per " + fileGT);
        }
    }
}
File.close(f);
print("--- FINITO! Processati " + count + " file. Apri il CSV! ---");