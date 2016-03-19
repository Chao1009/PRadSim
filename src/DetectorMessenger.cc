//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: DetectorMessenger.cc, 2012-08-01 $
// GEANT4 tag $Name: geant4-09-04-patch-02 $
// Developer: Chao Peng
// 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "DetectorMessenger.hh"

#include "DetectorConstruction.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::DetectorMessenger(
                                           DetectorConstruction* Det)
:Detector(Det)
{ 
  epsDir = new G4UIdirectory("/eps/");
  epsDir->SetGuidance("UI commands of this example");
  
  detDir = new G4UIdirectory("/eps/det/");
  detDir->SetGuidance("detector control");
       
  AbsMaterCmd = new G4UIcmdWithAString("/eps/det/setAbsMat",this);
  AbsMaterCmd->SetGuidance("Select Material of the Absorber.");
  AbsMaterCmd->SetParameterName("choice",false);
  AbsMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  
  TargetMaterCmd = new G4UIcmdWithAString("/eps/det/setTargetMat",this);
  TargetMaterCmd->SetGuidance("Select Material of the Target.");
  TargetMaterCmd->SetParameterName("choice",false);
  TargetMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
 
  CellMaterCmd = new G4UIcmdWithAString("/eps/det/setCellMat",this);
  CellMaterCmd->SetGuidance("Select Material of the Cell.");
  CellMaterCmd->SetParameterName("choice",false);
  CellMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  NeckMaterCmd = new G4UIcmdWithAString("/eps/det/setNeckMat",this);
  NeckMaterCmd->SetGuidance("Select Material of the Cell.");
  NeckMaterCmd->SetParameterName("choice", false);
  NeckMaterCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
   
  AbsThickCmd = new G4UIcmdWithADoubleAndUnit("/eps/det/setAbsThick",this);
  AbsThickCmd->SetGuidance("Set Thickness of the Absorber");
  AbsThickCmd->SetParameterName("Size",false);
  AbsThickCmd->SetRange("Size>=0.");
  AbsThickCmd->SetUnitCategory("Length");
  AbsThickCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  
  SizeXYCmd = new G4UIcmdWithADoubleAndUnit("/eps/det/setSizeXY",this);
  SizeXYCmd->SetGuidance("Set tranverse size of the calorimeter");
  SizeXYCmd->SetParameterName("Size",false);
  SizeXYCmd->SetRange("Size>0.");
  SizeXYCmd->SetUnitCategory("Length");    
  SizeXYCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  UpdateCmd = new G4UIcmdWithoutParameter("/eps/det/update",this);
  UpdateCmd->SetGuidance("Update calorimeter geometry.");
  UpdateCmd->SetGuidance("This command MUST be applied before \"beamOn\" ");
  UpdateCmd->SetGuidance("if you changed geometrical value(s).");
  UpdateCmd->AvailableForStates(G4State_Idle);
      
  MagFieldCmd = new G4UIcmdWithADoubleAndUnit("/eps/det/setField",this);  
  MagFieldCmd->SetGuidance("Define magnetic field.");
  MagFieldCmd->SetGuidance("Magnetic field will be in Z direction.");
  MagFieldCmd->SetParameterName("Bz",false);
  MagFieldCmd->SetUnitCategory("Magnetic flux density");
  MagFieldCmd->AvailableForStates(G4State_PreInit,G4State_Idle);  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::~DetectorMessenger()
{

  delete AbsMaterCmd; delete TargetMaterCmd; delete CellMaterCmd; 
  delete AbsThickCmd; delete NeckMaterCmd;
  delete SizeXYCmd;   delete UpdateCmd;
  delete MagFieldCmd;
  delete detDir;
  delete epsDir;  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == AbsMaterCmd )
   { Detector->SetAbsorberMaterial(newValue);}
   
  if( command == TargetMaterCmd )
   { Detector->SetTargetMaterial(newValue);}
  
  if( command == CellMaterCmd )
   { Detector->SetCellMaterial(newValue);}

  if( command == NeckMaterCmd )
   { Detector->SetNeckMaterial(newValue);}

  if( command == AbsThickCmd )
   { Detector->SetAbsorberThickness(AbsThickCmd
                                               ->GetNewDoubleValue(newValue));}
   
  if( command == SizeXYCmd )
   { Detector->SetCalorSizeXY(SizeXYCmd->GetNewDoubleValue(newValue));}

  if( command == UpdateCmd )
   { Detector->UpdateGeometry(); }

  if( command == MagFieldCmd )
   { Detector->SetMagField(MagFieldCmd->GetNewDoubleValue(newValue));}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
