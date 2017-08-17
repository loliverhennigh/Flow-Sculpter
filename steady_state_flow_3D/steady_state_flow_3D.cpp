/*  Lattice Boltzmann sample, written in C++, using the OpenLB
 *  library
 *
 *  Copyright (C) 2011-2013 Mathias J. Krause, Thomas Henn, Tim Dornieden
 *  E-mail contact: info@openlb.net
 *  The most recent release of OpenLB can be downloaded at
 *  <http://www.openlb.net/>
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this program; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA  02110-1301, USA.
 */

/* cylinder3d.cpp:
 * This example examines a steady flow past a cylinder placed in a channel.
 * The cylinder is offset somewhat from the center of the flow to make the
 * steady-state symmetrical flow unstable. At the inlet, a Poiseuille profile is
 * imposed on the velocity, whereas the outlet implements a Dirichlet pressure
 * condition set by p = 0.
 * Inspired by "Benchmark Computations of Laminar Flow Around
 * a Cylinder" by M.Schäfer and S.Turek. For high resolution, low
 * latticeU, and enough time to converge, the results for pressure drop, drag
 * and lift lie within the estimated intervals for the exact results.
 * An unsteady flow with Karman vortex street can be created by changing the
 * Reynolds number to Re=100.
 * It also shows the usage of the STL-reader and explains how
 * to set boundary conditions automatically.
 */


#include "olb3D.h"
#ifndef OLB_PRECOMPILED // Unless precompiled version is used,
#include "olb3D.hh"   // include full template code
#endif
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace olb::util;
using namespace std;

typedef double T;

typedef unsigned char byte;

#define DESCRIPTOR D3Q19Descriptor


// Parameters for the simulation setup
const T maxPhysT = 160.; // max. simulation time in s, SI unit

// Convergence params
const T interval = 5.0; // Time intervall in seconds for convergence check
const T epsilon = 1e-3; // Residuum for convergence check

// Vox file params
static int version;
static int depth, height, width;
static int size;
static byte *voxels = 0;
static float tx, ty, tz;
static float scale;

int read_binvox(string filespec)
{

  ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);

  //
  // read header
  //
  string line;
  *input >> line;  // #binvox
  if (line.compare("#binvox") != 0) {
    cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
    delete input;
    return 0;
  }
  *input >> version;
  cout << "reading binvox version " << version << endl;

  depth = -1;
  int done = 0;
  while(input->good() && !done) {
    *input >> line;
    if (line.compare("data") == 0) done = 1;
    else if (line.compare("dim") == 0) {
      *input >> depth >> height >> width;
    }
    else if (line.compare("translate") == 0) {
      *input >> tx >> ty >> tz;
    }
    else if (line.compare("scale") == 0) {
      *input >> scale;
    }
    else {
      cout << "  unrecognized keyword [" << line << "], skipping" << endl;
      char c;
      do {  // skip until end of line
        c = input->get();
      } while(input->good() && (c != '\n'));

    }
  }
  if (!done) {
    cout << "  error reading header" << endl;
    return 0;
  }
  if (depth == -1) {
    cout << "  missing dimensions in header" << endl;
    return 0;
  }

  size = width * height * depth;
  voxels = new byte[size];
  if (!voxels) {
    cout << "  error allocating memory" << endl;
    return 0;
  }

  //
  // read voxel data
  //
  byte value;
  byte count;
  int index = 0;
  int end_index = 0;
  int nr_voxels = 0;
  
  input->unsetf(ios::skipws);  // need to read every byte now (!)
  *input >> value;  // read the linefeed char

  while((end_index < size) && input->good()) {
    *input >> value >> count;

    if (input->good()) {
      end_index = index + count;
      if (end_index > size) return 0;
      for(int i=index; i < end_index; i++) voxels[i] = value;
      
      if (value) nr_voxels += count;
      index = end_index;
    }  // if file still ok
    
  }  // while

  input->close();
  cout << "  read " << nr_voxels << " voxels" << endl;

  return 1;

}

// Stores data from stl file in geometry in form of material numbers
void prepareGeometry( LBconverter<T> const& converter, IndicatorF3D<T>& indicator,
                      string binvox_filename, SuperGeometry3D<T>& superGeometry ) {

  OstreamManager clout( std::cout,"prepareGeometry" );
  clout << "Prepare Geometry ..." << std::endl;

  superGeometry.rename( 0,2,indicator );
  superGeometry.rename( 2,1,1,1,1 );
  superGeometry.clean();

  Vector<T,3> origin = superGeometry.getStatistics().getMinPhysR( 2 );
  origin[1] += converter.getLatticeL()/2.;
  origin[2] += converter.getLatticeL()/2.;

  Vector<T,3> extend = superGeometry.getStatistics().getMaxPhysR( 2 );
  extend[1] = extend[1]-origin[1]-converter.getLatticeL()/2.;
  extend[2] = extend[2]-origin[2]-converter.getLatticeL()/2.;

  // Set material number for inflow
  origin[0] = superGeometry.getStatistics().getMinPhysR( 2 )[0]-converter.getLatticeL();
  extend[0] = 2*converter.getLatticeL();
  IndicatorCuboid3D<T> inflow( extend,origin );
  superGeometry.rename( 2,3,inflow );

  // Set material number for outflow
  origin[0] = superGeometry.getStatistics().getMaxPhysR( 2 )[0]-converter.getLatticeL();
  extend[0] = 2*converter.getLatticeL();
  IndicatorCuboid3D<T> outflow( extend,origin );
  superGeometry.rename( 2,4,outflow );

  // Set material number for cylinder
  if (!read_binvox(binvox_filename)) {
    cout << "Error reading [" << binvox_filename << "]" << endl << endl;
    exit(1);
  }
  for ( int i = 0; i < depth; ++i ) {
    for ( int j = 0; j < height; ++j ) {
      for ( int k = 0; k < width; ++k ) {
        if ((char) (voxels[i+depth*j+depth*height*k] + '0') == '1') {
          superGeometry.set(0,i + depth/4,j + height/4,k + width/4) = 5;
        }
      }
    }
  }

  // Removes all not needed boundary voxels outside the surface
  superGeometry.clean();
  superGeometry.checkForErrors();

  superGeometry.print();

  clout << "Prepare Geometry ... OK" << std::endl;
}

// Set up the geometry of the simulation
void prepareLattice( SuperLattice3D<T,DESCRIPTOR>& sLattice,
                     LBconverter<T> const& converter,
                     Dynamics<T, DESCRIPTOR>& bulkDynamics,
                     sOnLatticeBoundaryCondition3D<T,DESCRIPTOR>& bc,
                     sOffLatticeBoundaryCondition3D<T,DESCRIPTOR>& offBc,
                     SuperGeometry3D<T>& superGeometry ) {

  OstreamManager clout( std::cout,"prepareLattice" );
  clout << "Prepare Lattice ..." << std::endl;

  const T omega = converter.getOmega();

  // Material=0 -->do nothing
  sLattice.defineDynamics( superGeometry, 0, &instances::getNoDynamics<T, DESCRIPTOR>() );

  // Material=1 -->bulk dynamics
  sLattice.defineDynamics( superGeometry, 1, &bulkDynamics );

  // Material=2 -->bounce back
  sLattice.defineDynamics( superGeometry, 2, &instances::getBounceBack<T, DESCRIPTOR>() );

  // Material=3 -->bulk dynamics (inflow)
  sLattice.defineDynamics( superGeometry, 3, &bulkDynamics );

  // Material=4 -->bulk dynamics (outflow)
  sLattice.defineDynamics( superGeometry, 4, &bulkDynamics );

  // Setting of the boundary conditions
  bc.addVelocityBoundary( superGeometry, 3, omega );
  bc.addPressureBoundary( superGeometry, 4, omega );

  // Material=5 -->bouzidi
  sLattice.defineDynamics(superGeometry, 5, &instances::getBounceBack<T, DESCRIPTOR>());
  //sLattice.defineDynamics( superGeometry, 5, &instances::getNoDynamics<T,DESCRIPTOR>() );
  //offBc.addZeroVelocityBoundary( superGeometry, 5, stlReader );

  // Initial conditions
  AnalyticalConst3D<T,T> rhoF( 1 );
  std::vector<T> velocity( 3,T() );
  AnalyticalConst3D<T,T> uF( velocity );

  // Initialize all values of distribution functions to their local equilibrium
  sLattice.defineRhoU( superGeometry, 1, rhoF, uF );
  sLattice.iniEquilibrium( superGeometry, 1, rhoF, uF );
  sLattice.defineRhoU( superGeometry, 3, rhoF, uF );
  sLattice.iniEquilibrium( superGeometry, 3, rhoF, uF );
  sLattice.defineRhoU( superGeometry, 4, rhoF, uF );
  sLattice.iniEquilibrium( superGeometry, 4, rhoF, uF );

  // Make the lattice ready for simulation
  sLattice.initialize();

  clout << "Prepare Lattice ... OK" << std::endl;
}

// Generates a slowly increasing inflow for the first iTMaxStart timesteps
void setBoundaryValues( SuperLattice3D<T, DESCRIPTOR>& sLattice,
                        LBconverter<T> const& converter, int iT,
                        SuperGeometry3D<T>& superGeometry ) {

  OstreamManager clout( std::cout,"setBoundaryValues" );

  // No of time steps for smooth start-up
  int iTmaxStart = converter.numTimeSteps( 20.0 );
  int iTupdate = 30;

  if ( iT%iTupdate == 0 && iT <= iTmaxStart ) {
    // Smooth start curve, sinus
    // SinusStartScale<T,int> StartScale(iTmaxStart, T(1));

    // Smooth start curve, polynomial
    PolynomialStartScale<T,int> StartScale( iTmaxStart, T( 1 ) );

    // Creates and sets the Poiseuille inflow profile using functors
    int iTvec[1] = {iT};
    T frac[1] = {};
    StartScale( frac,iTvec );
    std::vector<T> maxVelocity( 3,0 );
    maxVelocity[0] = 2.25*frac[0]*converter.getLatticeU();

    T distance2Wall = converter.getLatticeL()/2.;
    RectanglePoiseuille3D<T> poiseuilleU( superGeometry, 3, maxVelocity, distance2Wall, distance2Wall, distance2Wall );
    sLattice.defineU( superGeometry, 3, poiseuilleU );

    clout << "step=" << iT << "; maxVel=" << maxVelocity[0] << std::endl;
  }
}

// Computes the pressure drop between the voxels before and after the cylinder
void getResults( SuperLattice3D<T, DESCRIPTOR>& sLattice,
                 LBconverter<T>& converter, int iT,
                 SuperGeometry3D<T>& superGeometry, Timer<T>& timer,
                 bool converged ) {

  OstreamManager clout( std::cout,"getResults" );

  SuperVTMwriter3D<T> vtmWriter( "cylinder3d" );
  SuperLatticePhysVelocity3D<T, DESCRIPTOR> velocity( sLattice, converter );
  SuperLatticePhysPressure3D<T, DESCRIPTOR> pressure( sLattice, converter );
  vtmWriter.addFunctor( velocity );
  vtmWriter.addFunctor( pressure );

  const int vtkIter  = converter.numTimeSteps( .3 );
  const int statIter = converter.numTimeSteps( .1 );

  if ( iT==0 ) {
    // Writes the geometry, cuboid no. and rank no. as vti file for visualization
    SuperLatticeGeometry3D<T, DESCRIPTOR> geometry( sLattice, superGeometry );
    SuperLatticeCuboid3D<T, DESCRIPTOR> cuboid( sLattice );
    SuperLatticeRank3D<T, DESCRIPTOR> rank( sLattice );
    vtmWriter.write( geometry );
    vtmWriter.write( cuboid );
    vtmWriter.write( rank );

    vtmWriter.createMasterFile();
  }

  // Writes the vtk files
  if ( converged ) {
    vtmWriter.write( iT );

    SuperEuklidNorm3D<T, DESCRIPTOR> normVel( velocity );
    BlockLatticeReduction3D<T, DESCRIPTOR> planeReduction( normVel, 0, 0, -1 );
    BlockGifWriter<T> gifWriter;
    //gifWriter.write(planeReduction, 0, 0.7, iT, "vel"); //static scale
    gifWriter.write( planeReduction, iT, "vel" ); // scaled
  }

  // Gnuplot constructor (must be static!)
  static Gnuplot<T> gplot( "drag" );

  // write pdf at last time step
  if ( (iT == converter.numTimeSteps( maxPhysT )-1) || converged ) {
    // writes pdf
    gplot.writePDF();
  }

  // Writes output on the console
  if ( (iT%statIter == 0) || converged ) {
    // Timer console output
    timer.update( iT );
    timer.printStep();

    // Lattice statistics console output
    sLattice.getStatistics().print( iT,converter.physTime( iT ) );

    // Drag, lift, pressure drop
    AnalyticalFfromSuperF3D<T> intpolatePressure( pressure, true );
    SuperLatticePhysDrag3D<T,DESCRIPTOR> drag( sLattice, superGeometry, 5, converter );

    std::vector<T> point1V = superGeometry.getStatistics().getCenterPhysR( 5 );
    std::vector<T> point2V = superGeometry.getStatistics().getCenterPhysR( 5 );
    T point1[3] = {};
    T point2[3] = {};
    for ( int i = 0; i<3; i++ ) {
      point1[i] = point1V[i];
      point2[i] = point2V[i];
    }
    point1[0] = superGeometry.getStatistics().getMinPhysR( 5 )[0] - converter.getLatticeL();
    point2[0] = superGeometry.getStatistics().getMaxPhysR( 5 )[0] + converter.getLatticeL();

    T p1, p2;
    intpolatePressure( &p1,point1 );
    intpolatePressure( &p2,point2 );

    clout << "pressure1=" << p1;
    clout << "; pressure2=" << p2;

    T pressureDrop = p1-p2;
    clout << "; pressureDrop=" << pressureDrop;

    T dragA[3];
    int input1[0];
    drag( dragA, input1 );
    clout << "; drag=" << dragA[0] << "; lift=" << dragA[1] << endl;

    // set data for gnuplot: input={xValue, yValue(s), names (optional), position of key (optional)}
    gplot.setData( converter.physTime( iT ), {dragA[0]}, {"drag(openLB)"}, "bottom right" );

    // writes a png in one file for every timestep, if the file is open it can be used as a "liveplot"
    gplot.writePNG();

    SuperEuklidNorm3D<T, DESCRIPTOR> normVel( velocity );
    BlockLatticeReduction3D<T, DESCRIPTOR> planeReduction( normVel, 0, 0, -1 );
    BlockGifWriter<T> gifWriter;
    //gifWriter.write(planeReduction, 0, 0.7, iT, "vel"); //static scale
    gifWriter.write( planeReduction, iT, "vel" ); // scaled

    // every (iT%vtkIter) write an png of the plot
    if ( (iT%( vtkIter ) == 0) || converged ) {
      // writes pngs: input={name of the files (optional), x range for the plot (optional)}
      gplot.writePNG( iT, maxPhysT );
    }

  }
}

int main( int argc, char* argv[] ) {

  // === 1st Step: Initialization from xml ===
  if (argc!=2) {
    printf("need to give xml filename to run on \n");
    exit(1);
  }

  std::string fName(argv[1]);
  XMLreader config(fName);

  int simulation_size;
  string binvox_filename;
  string save_path;
  simulation_size = config["size"].get<int>();
  binvox_filename = config["binvox_name"].get<string>();
  save_path = config["save_path_3d"].get<string>();

  olbInit( &argc, &argv );
  singleton::directories().setOutputDir( save_path );
  OstreamManager clout( std::cout,"main" );
  // display messages from every single mpi process
  //clout.setMultiOutput(true);

  LBconverter<T> converter(
    ( int ) 3,                             // dim
    ( T )   1.28/simulation_size,          // latticeL_
    ( T )   0.02,                         // latticeU_
    ( T )   0.004,                          // charNu_
    ( T )   0.1,                           // charL_ = 1
    ( T )   0.2                            // charU_ = 1
  );
  converter.print();
  writeLogFile( converter, "cylinder3d" );


  // === 2nd Step: Prepare Geometry ===

  // Instantiation chamber the object will rest in
  Vector<T,3> origin ( 0.0,  0.0,  0.0);
  Vector<T,3> extend ( 3.84 - (2.5*converter.getLatticeL()), 1.92 - (2.5*converter.getLatticeL()), 1.92 - (2.5*converter.getLatticeL()));
  IndicatorCuboid3D<T> extendedDomain(extend, origin);

  // Instantiation of a cuboidGeometry with weights
#ifdef PARALLEL_MODE_MPI
  const int noOfCuboids = singleton::mpi().getSize();
#else
  const int noOfCuboids = 1;
#endif
  CuboidGeometry3D<T> cuboidGeometry( extendedDomain, converter.getLatticeL(), noOfCuboids );

  // Instantiation of a loadBalancer
  HeuristicLoadBalancer<T> loadBalancer( cuboidGeometry );

  // Instantiation of a superGeometry
  SuperGeometry3D<T> superGeometry( cuboidGeometry, loadBalancer, 2 );

  prepareGeometry( converter, extendedDomain, binvox_filename, superGeometry );

  // === 3rd Step: Prepare Lattice ===
  SuperLattice3D<T, DESCRIPTOR> sLattice( superGeometry );
  BGKdynamics<T, DESCRIPTOR> bulkDynamics( converter.getOmega(), instances::getBulkMomenta<T, DESCRIPTOR>() );

  // choose between local and non-local boundary condition
  sOnLatticeBoundaryCondition3D<T,DESCRIPTOR> sBoundaryCondition( sLattice );
  //createInterpBoundaryCondition3D<T,DESCRIPTOR>( sBoundaryCondition );
  createLocalBoundaryCondition3D<T,DESCRIPTOR>(sBoundaryCondition);

  sOffLatticeBoundaryCondition3D<T, DESCRIPTOR> sOffBoundaryCondition( sLattice );
  createBouzidiBoundaryCondition3D<T, DESCRIPTOR> ( sOffBoundaryCondition );

  prepareLattice( sLattice, converter, bulkDynamics, sBoundaryCondition, sOffBoundaryCondition, superGeometry );

  // === 4th Step: Main Loop with Timer ===
  util::ValueTracer<T> converge( converter.numTimeSteps(interval), epsilon );
  clout << "starting simulation..." << endl;
  Timer<T> timer( converter.numTimeSteps( maxPhysT ), superGeometry.getStatistics().getNvoxel() );
  timer.start();

  for ( int iT = 0; iT < converter.numTimeSteps( maxPhysT ); ++iT ) {

    // check if converged
    if ( converge.hasConverged() ) {
      clout << "Simulation converged." << endl;
      getResults( sLattice, converter, iT, superGeometry, timer, converge.hasConverged() );
      break;
    }

    // === 5th Step: Definition of Initial and Boundary Conditions ===
    setBoundaryValues( sLattice, converter, iT, superGeometry );

    // === 6th Step: Collide and Stream Execution ===
    sLattice.collideAndStream();

    // === 7th Step: Computation and Output of the Results ===
    getResults( sLattice, converter, iT, superGeometry, timer, converge.hasConverged() );
    converge.takeValue( sLattice.getStatistics().getAverageEnergy(), true );
  }

  timer.stop();
  timer.printSummary();
}

