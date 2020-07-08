#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

using namespace tensorflow;

const string PathToGraph = "/home/Test/lb1_2189_2.jpg";
const string PathToModel = "/home/Network";

int main()
{
    auto session = NewSession(SessionOptions());
    if (session == nullptr)
    {
        std::cout << "session error" << std::endl;
        return -1;
    }

    Status status;

    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "/home/Network/Network.pb", &graph_def);
    if (!status.ok())
    {
        std::cout << status.error_message() << std::endl;
        return -1;
    }

    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cout << status.error_message() << std::endl;
        return -1;
    }

    Tensor in1(DT_FLOAT, TensorShape());
    in1.scalar<float>()() = 1;

    Tensor in2(DT_BOOL, TensorShape());
    in2.scalar<bool>()() = false;

    std::vector<std::pair<string, Tensor>> inputs = {
            {"define_input/input_data:0", in1},
            {"define_input/training:0", in2},
    };
    std::vector<Tensor> outputs;

    status = session->Run(inputs, {"Network/reshape:0"}, {}, &outputs);
    if (!status.ok())
    {
        std::cout << status.error_message() << std::endl;
        return -1;
    }

    return 0;
}
