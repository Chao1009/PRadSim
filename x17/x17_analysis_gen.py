import os
import ROOT
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_hist_contents(root_hist):
    xfunc = np.vectorize(root_hist.GetBinCenter, otypes=[float])
    yfunc = np.vectorize(root_hist.GetBinContent, otypes=[float])
    bins = root_hist.GetNbinsX()
    return xfunc(np.arange(1, bins + 1, dtype=int)), yfunc(np.arange(1, bins + 1, dtype=int))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('analyze x17 simulation events')
    parser.add_argument('root_files', help='root files of simultion output, seperated by \",\"')

    args = parser.parse_args()

    rdf = ROOT.RDataFrame('T', args.root_files.split(','))

    ROOT.gInterpreter.Declare('''
        TRandom3 rng;

        int cluster_count(const ROOT::VecOps::RVec<double> &vals, double thres = 20.) {
            int count = 0;
            for (auto &val : vals) {
                if (val >= thres) { count ++; }
            }
            return count;
        }

        double sum_energy(const ROOT::VecOps::RVec<double> &vals, double thres = 20.) {
            double sum = 0.;
            for (auto &val : vals) {
                if (val > thres) { sum += val; }
            }
            return sum;
        }

        double sum_energy_crystal(const ROOT::VecOps::RVec<double> &vals,
                                  const ROOT::VecOps::RVec<double> &xs,
                                  const ROOT::VecOps::RVec<double> &ys,
                                  int Nhit) {
            double sum = 0.;
            for (int i = 0; i < Nhit; ++i) {
                if (std::abs(xs[i]) < 354.45 && std::abs(ys[i]) < 354.45) {
                    sum += vals[i];
                }
            }
            return sum;
        }

        double pair_inv_mass(const ROOT::VecOps::RVec<double> &xs,
                        const ROOT::VecOps::RVec<double> &ys,
                        const ROOT::VecOps::RVec<double> &zs,
                        const ROOT::VecOps::RVec<double> &ps,
                        const ROOT::VecOps::RVec<double> &phis,
                        double Eb = 3300.,
                        double target_pos = -300.,
                        double thres = 20.) {

            // need three particles
            if (ps.size() != 3) { return -1.; }

            // assuming the highest momentum particle is the scattered electron (should do PID in future analysis)
            // pick the e+ e- pair
            int i1 = 0, i2 = 1;
            if (ps[i1] < ps[i2]) {
                i1 = 1;
                i2 = 0;
            }
            if (ps[i1] > ps[2]) {
                i1 = 2;
            }

            // missing mass for e+e- pair
            double p_out[3] = {0., 0., 0.};
            double E_out = 0.;
            // merged pair
            double dx = xs[i1] - xs[i2];
            double dy = ys[i1] - ys[i2];
            if (std::sqrt(dx*dx + dy*dy) < 20.5*1.414) {
                return -1;
            }
            // build three momentum
            for (auto i : {i1,  i2}) {
                double x = xs[i];
                double y = ys[i];
                // geometrical cuts (in hycal crystal part)
                if (std::abs(xs[i]) > 354.45 || std::abs(ys[i]) > 354.45) {
                    return -1;
                }
                if (std::abs(xs[i]) < 61.5 && std::abs(ys[i]) < 61.5) {
                    return -1;
                }
                double z = zs[i] - target_pos;
                double r = std::sqrt(x*x + y*y + z*z);
                double p = ps[i]*(1. + rng.Gaus(0., 0.026/std::sqrt(ps[i]/1000.)));
                // add to out momentum sum
                p_out[0] += x/r*p;
                p_out[1] += y/r*p;
                p_out[2] += z/r*p;
                // add to out energy sum (MeV)
                E_out += std::sqrt(p*p + 0.511*0.511);
            }

            double dp2 = p_out[0]*p_out[0] + p_out[1]*p_out[1] + p_out[2]*p_out[2];
            double dE2 = E_out * E_out;
            // std::cout << E_out << ", " << p_out[0] << ", " << p_out[1] << ", " << p_out[2] << ", " << std::sqrt(dE2 - dp2) << std::endl;
            return std::sqrt(dE2 - dp2);
        }

        double inv_mass(const ROOT::VecOps::RVec<double> &ps, double Eb = 3300., double thres = 20.) {
            double W2 = Eb*Eb;
            double total_p = 0.;
            for (auto &p : ps) {
                if (p > thres) {
                    total_p += p*(1. + rng.Gaus(0., 0.));
                }
            }
            // return Eb - total_p;
            return std::sqrt(std::max(0., W2 - total_p*total_p));
        }
    ''')
    rdf = rdf.Define('Trgsum', 'sum_energy_crystal(VD.P, VD.X, VD.Y, VD.N)')

    counts = rdf.Count().GetValue()
    current = 50e-9/1.602e-19
    beamtime = counts/current
    print(counts, current, beamtime)
    rdf = rdf.Filter('Trgsum > 0.75*3300')
    # trigger rates
    print(rdf.Count().GetValue()/beamtime)
    rdf = rdf.Define('CN', 'cluster_count(VD.P)')
    # rdf = rdf.Filter('CN >= 2')
    # rdf = rdf.Filter('CN == 3')
    rdf = rdf.Define('W', 'pair_inv_mass(VD.X, VD.Y, VD.Z, VD.P, VD.Phi)')
    # rdf = rdf.Define('W', 'inv_mass(VD.P)')
    rdf = rdf.Filter('W > 0')
    rdf = rdf.Define('Wp', 'W+1')
    print(rdf.Count().GetValue())
    hist = rdf.Histo1D(ROOT.RDF.TH1DModel('invm', 'Invariant Mass; Counts / 2 MeV; W (MeV)', 40, 0, 80), 'Wp')
    print(get_hist_contents(hist))
    c = ROOT.TCanvas('signal_sum', '', 1920, 1080)
    hist.Draw('hist')
    c.SaveAs('X17_W_full.png')
    hist = rdf.Histo1D(ROOT.RDF.TH1DModel('trgsum', 'Crystal Edep Sum;Counts / 10 MeV;W (MeV)', 330, 0, 3300), 'Trgsum')
    c = ROOT.TCanvas('signal_sum', '', 1920, 1080)
    hist.Draw('hist')
    c.SaveAs('X17_trg_full.png')

