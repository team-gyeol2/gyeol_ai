#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/wifi-module.h"

#include <arpa/inet.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavAdhocLogging");

namespace
{

// ── 기존 로깅 변수 ────────────────────────────────────────────────────────────
std::ofstream g_rssiStream;
std::ofstream g_rttStream;
std::ofstream g_posStream;
std::ofstream g_mlStream;   // ML 연동 로그
uint32_t      g_pingReplies  = 0;
double        g_rttSumMs     = 0.0;
uint32_t      g_rssiSamples  = 0;
double        g_rssiSumDbm   = 0.0;

// ── ML 서버 연동 변수 ─────────────────────────────────────────────────────────
int    g_mlSocket = -1;                                  // TCP 소켓 (-1 = 오프라인)
double g_latSum   = 0.0;                                 // 누적 지연 시간
int    g_latCount = 0;                                   // 응답 횟수

// (src_id, dst_id) → 경로 손실 기반 RSSI (dBm)
std::map<std::pair<uint32_t, uint32_t>, double> g_linkRssi;

// ── 전파 파라미터 (wifiPhy 설정과 동일) ──────────────────────────────────────
constexpr double TX_POWER_DBM  = 16.0;
constexpr double REF_LOSS_DB   = 46.6777;
constexpr double PATH_EXPONENT = 2.7;


// ── 헬퍼 ─────────────────────────────────────────────────────────────────────
std::string
FormatVector(const Vector& p)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "(" << p.x << ", " << p.y << ", " << p.z << ")";
    return oss.str();
}


// ── ML 서버 연결 ──────────────────────────────────────────────────────────────
bool
ConnectToMlServer(const std::string& host = "127.0.0.1", uint16_t port = 9000)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        std::cerr << "[ML] socket() 실패" << std::endl;
        return false;
    }

    // 수신 타임아웃 500ms
    struct timeval tv{0, 500000};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        close(sock);
        std::cout << "[ML] Python 서버 없음 — 오프라인 모드로 실행" << std::endl;
        return false;
    }

    g_mlSocket = sock;
    std::cout << "[ML] 연결 성공: " << host << ":" << port << std::endl;
    return true;
}


// ── 전 쌍 RSSI 계산 (경로 손실 모델 기반) ───────────────────────────────────
void
ComputeLinkRssi(NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        for (uint32_t j = i + 1; j < nodes.GetN(); ++j)
        {
            auto pos_i = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
            auto pos_j = nodes.Get(j)->GetObject<MobilityModel>()->GetPosition();
            double dist     = std::max(CalculateDistance(pos_i, pos_j), 1.0);
            double pathLoss = REF_LOSS_DB + 10.0 * PATH_EXPONENT * std::log10(dist);
            g_linkRssi[{i, j}] = TX_POWER_DBM - pathLoss;
        }
    }
}


// ── NS-3 → Python JSON 직렬화 ─────────────────────────────────────────────────
std::string
BuildJson(NodeContainer nodes, double t)
{
    std::ostringstream j;
    j << std::fixed << std::setprecision(3);

    j << "{\"t\":" << t << ",\"uavs\":[";
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        auto pos = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
        if (i) j << ",";
        j << "{\"id\":" << i
          << ",\"x\":"  << pos.x
          << ",\"y\":"  << pos.y
          << ",\"z\":"  << pos.z << "}";
    }

    j << "],\"links\":[";
    bool first = true;
    for (auto& kv : g_linkRssi)
    {
        if (!first) j << ",";
        j << "{\"src\":" << kv.first.first
          << ",\"dst\":" << kv.first.second
          << ",\"rssi\":" << kv.second << "}";
        first = false;
    }
    j << "]}";
    return j.str();
}


// ── 응답 파싱: relay ID 추출 ──────────────────────────────────────────────────
int
ParseRelay(const std::string& resp)
{
    auto pos = resp.find("\"relay\":");
    if (pos == std::string::npos) return -1;
    try { return std::stoi(resp.substr(pos + 8)); }
    catch (...) { return -1; }
}


// ── 응답 파싱: 위치 보정 추출 ─────────────────────────────────────────────────
bool
ParseCorrection(const std::string& resp, int& uavId, double& dx, double& dy)
{
    // "correction": null 이면 스킵
    auto cpos = resp.find("\"correction\":");
    if (cpos == std::string::npos) return false;
    if (resp.find("null", cpos) != std::string::npos &&
        resp.find("null", cpos) < cpos + 20) return false;

    auto idPos = resp.find("\"uav_id\":", cpos);
    auto dxPos = resp.find("\"dx\":", cpos);
    auto dyPos = resp.find("\"dy\":", cpos);
    if (idPos == std::string::npos || dxPos == std::string::npos ||
        dyPos == std::string::npos) return false;

    try
    {
        uavId = std::stoi(resp.substr(idPos + 9));
        dx    = std::stod(resp.substr(dxPos + 5));
        dy    = std::stod(resp.substr(dyPos + 5));
        return true;
    }
    catch (...) { return false; }
}


// ── 매 1초 실행: 링크 상태 전송 & ML 결정 수신 ───────────────────────────────
void
SendLinkState(NodeContainer nodes)
{
    ComputeLinkRssi(nodes);
    const double now = Simulator::Now().GetSeconds();
    const std::string json = BuildJson(nodes, now);

    if (g_mlSocket < 0)
    {
        // 오프라인 모드: JSON만 로그에 기록
        std::cout << "[ML-offline] t=" << now << "s" << std::endl;
        if (g_mlStream.is_open())
            g_mlStream << now << ",offline,0,\"\"\n";
    }
    else
    {
        std::string msg = json + "\n";

        // 전송 + 지연 측정
        auto t0   = std::chrono::steady_clock::now();
        ssize_t sent = send(g_mlSocket, msg.c_str(), msg.size(), 0);

        if (sent < 0)
        {
            std::cerr << "[ML] 전송 실패" << std::endl;
        }
        else
        {
            char    buf[4096] = {};
            ssize_t n         = recv(g_mlSocket, buf, sizeof(buf) - 1, 0);
            auto    t1        = std::chrono::steady_clock::now();
            double  latMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (n > 0)
            {
                std::string resp(buf, n);
                g_latSum += latMs;
                ++g_latCount;

                std::cout << std::fixed << std::setprecision(2)
                          << "[ML] t=" << now << "s  lat=" << latMs
                          << "ms  " << resp << std::endl;

                // relay 결정 적용
                int relayId = ParseRelay(resp);
                if (relayId >= 0)
                    std::cout << "[ML] → Relay: UAV" << relayId << std::endl;

                // 위치 보정 적용
                int corrId = -1; double dx = 0, dy = 0;
                if (ParseCorrection(resp, corrId, dx, dy) &&
                    corrId >= 0 && corrId < (int)nodes.GetN())
                {
                    auto mob = nodes.Get(corrId)
                                   ->GetObject<ConstantVelocityMobilityModel>();
                    auto pos = mob->GetPosition();
                    mob->SetPosition(Vector(pos.x + dx, pos.y + dy, pos.z));
                    std::cout << "[ML] → UAV" << corrId
                              << " 위치 보정 (" << dx << ", " << dy << ")"
                              << std::endl;
                }

                // ML 로그 기록
                if (g_mlStream.is_open())
                    g_mlStream << std::fixed << std::setprecision(3)
                               << now << "," << latMs << ","
                               << relayId << ",\"" << resp << "\"\n";
            }
            else
            {
                std::cerr << "[ML] 응답 없음 (타임아웃)" << std::endl;
                if (g_mlStream.is_open())
                    g_mlStream << now << "," << latMs << ",-1,\"timeout\"\n";
            }
        }
    }

    // 다음 호출 예약 (시뮬레이터가 종료 시간 이후는 자동으로 무시함)
    Simulator::Schedule(Seconds(1.0), &SendLinkState, nodes);
}


// ── 위치 로깅 (기존) ──────────────────────────────────────────────────────────
void
LogPositions(NodeContainer nodes, Time interval)
{
    const double now = Simulator::Now().GetSeconds();
    std::cout << "[POS " << now << "s]";
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        const auto mobility = nodes.Get(i)->GetObject<MobilityModel>();
        const auto position = mobility->GetPosition();
        std::cout << " UAV" << i << "=" << FormatVector(position);
        if (g_posStream.is_open())
        {
            g_posStream << std::fixed << std::setprecision(6) << now << "," << i << ","
                        << position.x << "," << position.y << "," << position.z
                        << std::endl;
        }
    }
    std::cout << std::endl;

    if (Simulator::Now() + interval <= Simulator::GetMaximumSimulationTime())
        Simulator::Schedule(interval, &LogPositions, nodes, interval);
}


// ── RSSI 모니터 (기존) ────────────────────────────────────────────────────────
void
MonitorSniffRx(std::string context,
               Ptr<const Packet> packet,
               uint16_t channelFreqMhz,
               WifiTxVector txVector,
               MpduInfo aMpdu,
               SignalNoiseDbm signalNoise,
               uint16_t staId)
{
    const double now = Simulator::Now().GetSeconds();
    g_rssiSamples++;
    g_rssiSumDbm += signalNoise.signal;

    if (g_rssiStream.is_open())
    {
        g_rssiStream << std::fixed << std::setprecision(6) << now << "," << context << ","
                     << packet->GetSize() << "," << channelFreqMhz << ","
                     << signalNoise.signal << "," << signalNoise.noise << ","
                     << staId << std::endl;
    }

    std::cout << std::fixed << std::setprecision(3) << "[RSSI] t=" << now
              << "s signal=" << signalNoise.signal << " dBm noise=" << signalNoise.noise
              << " dBm size=" << packet->GetSize() << "B" << std::endl;
}


// ── Ping RTT (기존) ───────────────────────────────────────────────────────────
void
PingRtt(std::string context, uint16_t seqNo, Time rtt)
{
    const double now   = Simulator::Now().GetSeconds();
    const double rttMs = rtt.GetMilliSeconds();
    g_pingReplies++;
    g_rttSumMs += rttMs;

    if (g_rttStream.is_open())
    {
        g_rttStream << std::fixed << std::setprecision(6) << now << "," << seqNo << ","
                    << rttMs << "," << context << std::endl;
    }

    std::cout << std::fixed << std::setprecision(3) << "[RTT] t=" << now
              << "s seq=" << seqNo << " rtt=" << rttMs << " ms" << std::endl;
}

} // namespace


// ── main ──────────────────────────────────────────────────────────────────────
int
main(int argc, char* argv[])
{
    uint32_t    numUavs      = 5;
    double      spacing      = 25.0;
    double      altitude     = 30.0;
    double      speed        = 1.5;
    double      simTime      = 20.0;
    uint32_t    pingCount    = 10;
    double      pingInterval = 1.0;
    bool        enablePcap   = false;
    std::string mlHost       = "127.0.0.1";
    uint16_t    mlPort       = 9000;

    CommandLine cmd(__FILE__);
    cmd.AddValue("numUavs",      "Number of UAV nodes",                     numUavs);
    cmd.AddValue("spacing",      "Initial x-axis spacing (m)",              spacing);
    cmd.AddValue("altitude",     "Initial UAV altitude (m)",                altitude);
    cmd.AddValue("speed",        "Forward speed (m/s)",                     speed);
    cmd.AddValue("simTime",      "Simulation time (s)",                     simTime);
    cmd.AddValue("pingCount",    "Number of ping packets",                  pingCount);
    cmd.AddValue("pingInterval", "Ping interval (s)",                       pingInterval);
    cmd.AddValue("enablePcap",   "Enable PCAP trace",                       enablePcap);
    cmd.AddValue("mlHost",       "ML server IP (default 127.0.0.1)",        mlHost);
    cmd.AddValue("mlPort",       "ML server port (default 9000)",           mlPort);
    cmd.Parse(argc, argv);

    if (numUavs < 2)
        NS_ABORT_MSG("numUavs must be at least 2");

    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    // ── CSV 파일 열기 ─────────────────────────────────────────────────────────
    g_rssiStream.open("uav-rssi.csv");
    g_rttStream.open("uav-rtt.csv");
    g_posStream.open("uav-pos.csv");
    g_mlStream.open("uav-ml.csv");

    g_rssiStream << "time_s,context,packet_size_bytes,channel_mhz,"
                    "signal_dbm,noise_dbm,sta_id\n";
    g_rttStream  << "time_s,seq_no,rtt_ms,context\n";
    g_posStream  << "time_s,uav_id,x,y,z\n";
    g_mlStream   << "time_s,latency_ms,relay_id,response\n";

    // ── WiFi 노드 생성 ────────────────────────────────────────────────────────
    NodeContainer nodes;
    nodes.Create(numUavs);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",    StringValue("ErpOfdmRate24Mbps"),
                                 "ControlMode", StringValue("ErpOfdmRate6Mbps"));

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                   "Exponent",      DoubleValue(PATH_EXPONENT),
                                   "ReferenceLoss", DoubleValue(REF_LOSS_DB));

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(TX_POWER_DBM));
    wifiPhy.Set("TxPowerEnd",   DoubleValue(TX_POWER_DBM));
    wifiPhy.Set("RxGain",       DoubleValue(0.0));
    wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // ── 이동성 설정 ───────────────────────────────────────────────────────────
    MobilityHelper mobility;
    auto positions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < numUavs; ++i)
        positions->Add(Vector(i * spacing, 0.0, altitude));
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(nodes);

    for (uint32_t i = 0; i < numUavs; ++i)
    {
        auto model = nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        const double lateral = (i % 2 == 0) ? 0.4 : -0.4;
        model->SetVelocity(Vector(speed, lateral, 0.0));
    }

    // ── 라우팅 & IP ───────────────────────────────────────────────────────────
    OlsrHelper              olsr;
    Ipv4StaticRoutingHelper staticRouting;
    Ipv4ListRoutingHelper   listRouting;
    listRouting.Add(olsr, 10);
    listRouting.Add(staticRouting, 0);

    InternetStackHelper internet;
    internet.SetRoutingHelper(listRouting);
    internet.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    // ── Ping 앱 ───────────────────────────────────────────────────────────────
    PingHelper ping(interfaces.GetAddress(numUavs - 1));
    ping.SetAttribute("Count",       UintegerValue(pingCount));
    ping.SetAttribute("Interval",    TimeValue(Seconds(pingInterval)));
    ping.SetAttribute("VerboseMode", EnumValue(Ping::VerboseMode::SILENT));
    ApplicationContainer pingApp = ping.Install(nodes.Get(0));
    pingApp.Start(Seconds(1.0));
    pingApp.Stop(Seconds(simTime - 0.5));

    // ── 트레이스 연결 ─────────────────────────────────────────────────────────
    Config::Connect("/NodeList/0/ApplicationList/*/$ns3::Ping/Rtt",
                    MakeCallback(&PingRtt));

    const std::string rxPath =
        "/NodeList/" + std::to_string(numUavs - 1) +
        "/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx";
    Config::Connect(rxPath, MakeCallback(&MonitorSniffRx));

    FlowMonitorHelper flowMonitorHelper;
    auto monitor = flowMonitorHelper.InstallAll();

    if (enablePcap)
        wifiPhy.EnablePcap("uav-adhoc", devices);

    // ── ML 서버 연결 시도 ─────────────────────────────────────────────────────
    ConnectToMlServer(mlHost, mlPort);

    // ── 시뮬레이션 이벤트 예약 ────────────────────────────────────────────────
    Simulator::Schedule(Seconds(0.5), &LogPositions, nodes, Seconds(1.0));
    Simulator::Schedule(Seconds(1.0), &SendLinkState, nodes); // ML 연동 시작

    std::cout << "UAV Ad-hoc 시뮬레이션 시작 (" << numUavs << "대)" << std::endl;
    std::cout << "Source: UAV0 (" << interfaces.GetAddress(0)
              << ")  Dest: UAV" << (numUavs - 1)
              << " (" << interfaces.GetAddress(numUavs - 1) << ")" << std::endl;
    std::cout << "ML 서버: " << (g_mlSocket >= 0 ? "연결됨" : "오프라인") << std::endl;

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();

    // ── 요약 출력 ─────────────────────────────────────────────────────────────
    const double sent     = static_cast<double>(pingCount);
    const double received = static_cast<double>(g_pingReplies);
    const double plr      = sent > 0.0 ? ((sent - received) / sent) * 100.0 : 0.0;
    const double avgRtt   = received > 0.0 ? g_rttSumMs / received : 0.0;
    const double avgRssi  = g_rssiSamples > 0 ? g_rssiSumDbm / g_rssiSamples : 0.0;
    const double avgLat   = g_latCount > 0 ? g_latSum / g_latCount : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Ping 전송: "        << pingCount       << std::endl;
    std::cout << "Ping 수신: "        << g_pingReplies   << std::endl;
    std::cout << "PLR: "              << plr             << " %" << std::endl;
    std::cout << "평균 RTT: "         << avgRtt          << " ms" << std::endl;
    std::cout << "평균 RSSI: "        << avgRssi         << " dBm" << std::endl;
    if (g_latCount > 0)
        std::cout << "ML 평균 응답 지연: " << avgLat << " ms ("
                  << g_latCount << "회)" << std::endl;

    // ── 정리 ─────────────────────────────────────────────────────────────────
    g_rssiStream.close();
    g_rttStream.close();
    g_posStream.close();
    g_mlStream.close();

    if (g_mlSocket >= 0)
        close(g_mlSocket);

    Simulator::Destroy();
    return 0;
}
