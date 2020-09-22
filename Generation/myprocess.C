/*
This macro shows how to compute jet energy scale.
root -l examples/Example4.C'("delphes_output.root", "plots.root")'
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

class ExRootResult;
class ExRootTreeReader;

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader,  const char *outputFile_part)
{
  TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchEvent = treeReader->UseBranch("Event");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
  TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
  TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
  
  Long64_t allEntries = treeReader->GetEntries();
  ofstream myfile_det;
  ofstream myfile_part;

  cout << "** Chain contains " << allEntries << " events" << endl;

  Jet  *genjet;
  Jet *jet;
  GenParticle *muparticle;
  GenParticle *genparticle;
  GenParticle *motherparticle;
  TObject *object;
  Photon *photon;
  Track *track;
  Tower *tower;
  
  TLorentzVector genJetMomentum;
  TLorentzVector jetMomentum;
  TLorentzVector myMomentum;
  
  TLorentzVector incomingelectron;
  TLorentzVector incomingpositron;
  TLorentzVector outgoingphoton;
  
  Long64_t entry;

  Int_t i, j;

  myfile_part.open (outputFile_part);

  // Loop over all events
  for(entry = 0; entry < allEntries; ++entry)
  {
    //if (entry > 10000) break;
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    HepMCEvent *event = (HepMCEvent*) branchEvent -> At(0);
    //std::cout << "weight : " << event->Weight << std::endl;
    
    if(entry%500 == 0) cout << "Event number: "<< entry <<endl;

    myfile_part << event->Weight << " ";

    //First, let's find sqrt(hat(s))
    for(j = 0; j < branchParticle->GetEntriesFast(); ++j)
      {
        genparticle = (GenParticle*) branchParticle->At(j);
	if (abs(genparticle->PID)!=11 && genparticle->PID!=22) continue;
	//std::cout << " " << j << " " << genparticle->PID << " " << genparticle->Status << " " << motherparticle->Status << std::endl;  
	if (genparticle->M1 < 0){
	   if (genparticle->PID==11) incomingelectron = genparticle->P4();
	   if (genparticle->PID==-11) incomingpositron = genparticle->P4();
	}
	if (genparticle->M1 < 0) continue;
	motherparticle = (GenParticle*) branchParticle->At(genparticle->M1);
	if (motherparticle->Status==2) continue;
	if (genparticle->PID==22) outgoingphoton = genparticle->P4();
      }

    myfile_part << (incomingelectron+incomingpositron).M() << " " << (incomingelectron+incomingpositron-outgoingphoton).M() << " " << outgoingphoton.Pt() << " " << outgoingphoton.Eta() << " " << outgoingphoton.Phi() << " ";

    TLorentzVector highestpTphoton = TLorentzVector(0.1,0.1,0.1,0.1);
    for(i=0; i < branchEFlowPhoton->GetEntriesFast(); ++i){
      photon = (Photon*) branchEFlowPhoton->At(i);
      if (photon->P4().Pt() > highestpTphoton.Pt()) highestpTphoton = photon->P4();
      //std::cout << "    " << photon->P4().Eta() << " " << photon->P4().Pt() << " " << outgoingphoton.Eta() << " " << outgoingphoton.Pt() << " " << photon->SumPt << std::endl;
    }

    TLorentzVector visibleenergy = TLorentzVector(0.,0.,0.,0.);
    for(i=0; i < branchEFlowPhoton->GetEntriesFast(); ++i){
      photon = (Photon*) branchEFlowPhoton->At(i);
      myMomentum = photon->P4();
      visibleenergy+=myMomentum;
    }
    for(i=0; i < branchEFlowTrack->GetEntriesFast(); ++i){
      track = (Track*) branchEFlowTrack->At(i);
      myMomentum = track->P4();
      visibleenergy+=myMomentum;
    }
    for(i=0; i < branchEFlowNeutralHadron->GetEntriesFast(); ++i){
      tower = (Tower*) branchEFlowNeutralHadron->At(i);
      myMomentum = tower->P4();
      visibleenergy+=myMomentum;
    }

    myfile_part << visibleenergy.M() << " " << (visibleenergy-highestpTphoton).M() << " " << highestpTphoton.Pt() << " " << highestpTphoton.Eta() << " " << highestpTphoton.Phi() << " ";
    myfile_part << "J ";
    
    for(i = 0; i < branchJet->GetEntriesFast(); ++i){
      jet = (Jet*) branchJet->At(i);
      jetMomentum = jet->P4();
      myfile_part << i << " " << jet->PT << " " << jet->Eta << " " << jet->Phi << " " << jetMomentum.M() << " " << jet->BTag << " " << jet->Tau[0] << " " << jet->Tau[1] << " " << jet->Tau[2] << " " << jet->Tau[3] << " " << jet->Tau[4] << " ";
    }

    myfile_part << "P ";
    for(i=0; i < branchEFlowPhoton->GetEntriesFast(); ++i){
      photon = (Photon*) branchEFlowPhoton->At(i);
      myMomentum = photon->P4();
      myfile_part << i << " " << myMomentum.Pt() << " " << myMomentum.Eta() << " " << myMomentum.Phi() << " 22 ";
    }
    for(i=0; i < branchEFlowTrack->GetEntriesFast(); ++i){
      track = (Track*) branchEFlowTrack->At(i);
      myMomentum = track->P4();
      visibleenergy+=myMomentum;
      if (track->Charge > 0) myfile_part << i+branchEFlowPhoton->GetEntriesFast() << " " << myMomentum.Pt() << " " << myMomentum.Eta() << " " << myMomentum.Phi() << " 211 ";
      else myfile_part << i+branchEFlowPhoton->GetEntriesFast() << " " << myMomentum.Pt() << " " << myMomentum.Eta() << " " << myMomentum.Phi() << " -211 ";
    }
    for(i=0; i < branchEFlowNeutralHadron->GetEntriesFast(); ++i){
      tower = (Tower*) branchEFlowNeutralHadron->At(i);
      myMomentum = tower->P4();
      visibleenergy+=myMomentum;
      myfile_part << i+branchEFlowPhoton->GetEntriesFast()+branchEFlowNeutralHadron->GetEntriesFast() << " " << myMomentum.Pt() << " " << myMomentum.Eta() << " " << myMomentum.Phi() << " 2112 ";
    }
    
    myfile_part << std::endl;  

  }
}
//------------------------------------------------------------------------------

void myprocess(const char *inputFile, const char *outputFile_part)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalyseEvents(treeReader,outputFile_part);

  cout << "** Exiting..." << endl;

  delete result;
  delete treeReader;
  delete chain;
}

//------------------------------------------------------------------------------
