/* 将s1g-test-tim-raw.cc的所有include放在这里 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/internet-module.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include "ns3/rps.h"
#include <utility>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace ns3;
using namespace std;

// 直线AP布置配置结构
struct LinearAPConfig {
    double startX = 0.0;        
    double startY = 0.0;        
    double apSpacing = 100.0;   
    double lineAngle = 0.0;     
    uint32_t numAPs = 3;        
    double staRadius = 50.0;    
};

class LinearAPDeployment {
private:
    LinearAPConfig config;
    std::vector<Vector> apPositions;
    
public:
    LinearAPDeployment(const LinearAPConfig& cfg) : config(cfg) {
        calculateAPPositions();
    }
    
    void calculateAPPositions() {
        apPositions.clear();
        
        for (uint32_t i = 0; i < config.numAPs; i++) {
            double x = config.startX + i * config.apSpacing * cos(config.lineAngle);
            double y = config.startY + i * config.apSpacing * sin(config.lineAngle);
            apPositions.push_back(Vector(x, y, 0.0));
        }
    }
    
    void ConfigureAPMobility(NodeContainer& apNodes) {
        if (apNodes.GetN() != config.numAPs) {
            NS_FATAL_ERROR("AP节点数量与配置不匹配");
        }
        
        MobilityHelper mobility;
        Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
        
        // 添加每个AP的位置
        for (uint32_t i = 0; i < config.numAPs; i++) {
            positionAlloc->Add(apPositions[i]);
            std::cout << "AP " << i << " 位置: (" 
                      << apPositions[i].x << ", " << apPositions[i].y << ")" << std::endl;
        }
        
        mobility.SetPositionAllocator(positionAlloc);
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(apNodes);
    }
    
    void ConfigureSTAMobility(NodeContainer& staNodes, uint32_t stasPerAP) {
        if (staNodes.GetN() != config.numAPs * stasPerAP) {
            NS_FATAL_ERROR("STA节点总数与配置不匹配");
        }
        
        MobilityHelper mobility;
        Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
        
        uint32_t staIndex = 0;
        for (uint32_t apIdx = 0; apIdx < config.numAPs; apIdx++) {
            Vector apPos = apPositions[apIdx];
            
            // 每个AP周围的STA均匀分布在圆周上
            for (uint32_t staIdx = 0; staIdx < stasPerAP; staIdx++) {
                double angle = 2.0 * M_PI * staIdx / stasPerAP;
                double radius = config.staRadius * (0.5 + 0.5 * (double)rand() / RAND_MAX);
                
                double x = apPos.x + radius * cos(angle);
                double y = apPos.y + radius * sin(angle);
                
                positionAlloc->Add(Vector(x, y, 0.0));
                std::cout << "STA " << staIndex << " (属于AP " << apIdx << ") 位置: (" 
                          << x << ", " << y << ")" << std::endl;
                staIndex++;
            }
        }
        
        mobility.SetPositionAllocator(positionAlloc);
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(staNodes);
    }
};

/* 
 * 接下来粘贴s1g-test-tim-raw.cc的其他代码，
 * 但是将mobility配置部分替换为上面的LinearAPDeployment使用
 * 
 * 在main函数中找到原来的mobility配置代码（大约第962-978行），替换为：
 */

// 在main函数中替换mobility配置的代码：
void ConfigureLinearAPMobility(NodeContainer& wifiApNode, NodeContainer& wifiStaNode, string rho) {
    std::cout << "使用直线AP布置" << std::endl;
    
    LinearAPConfig apConfig;
    apConfig.startX = 0.0;
    apConfig.startY = 0.0;
    apConfig.apSpacing = 200.0;  // AP间距200米
    apConfig.lineAngle = 0.0;    // 水平布置
    apConfig.numAPs = wifiApNode.GetN();
    apConfig.staRadius = std::stoi(rho, nullptr, 0);
    
    LinearAPDeployment deployment(apConfig);
    deployment.ConfigureAPMobility(wifiApNode);
    
    uint32_t stasPerAP = wifiStaNode.GetN() / wifiApNode.GetN();
    deployment.ConfigureSTAMobility(wifiStaNode, stasPerAP);
}

/* 
 * 然后将s1g-test-tim-raw.cc的完整main函数代码粘贴到这里，
 * 但是将第962-978行的mobility配置代码替换为：
 *
 * ConfigureLinearAPMobility(wifiApNode, wifiStaNode, config.rho);
 *
 */
