#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

int main(){
    int input[3],output[3];
    std::vector<std::vector<float> > data;
    std::vector<unsigned char> data_label;
    std::ifstream in("network.bin",std::ios::binary);
    if(!in)
        return false;
    unsigned int i,j;
    in.read((char*)&input[0],sizeof(input[0])*3);
    in.read((char*)&output[0],sizeof(output[0])*3);
    in.read((char*)&i,4);
    in.read((char*)&j,4);
    data_label.resize(i);
    in.read((char*)&data_label[0],sizeof(unsigned char)*i);
    data.resize(i);
    cout<<i<<' '<<j<<endl;

    std::ofstream data_out("data.csv");
    for(unsigned int k = 0;k < i;++k)
    {
        data[k].resize(j);
        in.read((char*)&data[k][0],sizeof(float)*j);

        data_out << int(data_label[k]) << ' ';
        for (unsigned int n = 0; n < j; n++)
            data_out << data[k][n] << ' ';
        data_out << endl;
    }
    data_out.close();
    //for (int n = 0; n < i; n++)
    //    cout << int(data_label[n]);
    return 0;
}
