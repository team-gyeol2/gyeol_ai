/**
 * uav-multi-scenario.cc
 * ─────────────────────
 * 5개 시나리오 NS-3 실 시뮬레이션 (멀티링크 측정)
 *
 * 5개 UAV의 모든 링크 쌍(C(5,2)=10)에 대해
 * RSSI, PLR, 거리, SNR, 링크 상태를 0.25s 단위로 측정·기록한다.
 * RSSI: MonitorSniffRx (실제 PHY 측정)
 * PLR:  RSSI 기반 파생 추정 (Python 데이터셋과 동일 방식)
 *
 * 시나리오:
 *   0: corridor_baseline  — 동쪽 방향 직선 이동
 *   1: relay_stretch      — 동서 발산 (relay UAV2 정지)
 *   2: multi_disconnect_fast — 다중 동시 고속 이탈
 *   3: cluster_split_2    — 군집 2분리
 *   4: relay_failure      — relay(UAV2) 자체 이탈
 *
 * 실행:
 *   ./ns3 run "uav-multi-scenario --scenario=0"
 *
 * 출력:
 *   uav_ns3_<scenario_id>.csv
 */

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/wifi-module.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavMultiScenario");

// ── 상수 ──────────────────────────────────────────────────────────────────────
static const uint32_t NUM_UAVS  = 5;
static const double   TIME_STEP = 0.25;   // 측정 간격 (s)
static const double   NOISE_DBM = -95.0;  // 802.11g 노이즈 플로어

// 링크 상태 임계값 (Python 코드와 동일)
static const double HEALTHY_RSSI_MIN    = -78.0;
static const double DEGRADED_RSSI_MIN   = -85.0;
static const double HEALTHY_PLR_MAX     =   5.0;
static const double DEGRADED_PLR_MAX    =  20.0;

// ── 시나리오 정의 ─────────────────────────────────────────────────────────────
struct ScenarioConfig
{
    std::string id;
    double      duration_s;
    double      altitude_m;
    std::array<Vector, NUM_UAVS> init_pos;   // (x, y, z)
    std::array<Vector, NUM_UAVS> velocity;   // (vx, vy, vz)
};

static const std::array<ScenarioConfig, 5> SCENARIOS = {{
    // 0: corridor_baseline
    {
        "corridor_baseline", 60.0, 10.0,
        {{ Vector(45,75,10), Vector(45,45,10), Vector(60,60,10),
           Vector(75,45,10), Vector(75,75,10) }},
        {{ Vector(5,-0.5,0), Vector(5,0.5,0), Vector(5,0,0),
           Vector(5,-0.5,0), Vector(5,0.5,0) }}
    },
    // 1: relay_stretch
    {
        "relay_stretch", 20.0, 10.0,
        {{ Vector(95,70,10), Vector(95,50,10), Vector(95,60,10),
           Vector(95,65,10), Vector(95,55,10) }},
        {{ Vector(6,0.5,0), Vector(6,-0.5,0), Vector(0,0,0),
           Vector(-6,0.5,0), Vector(-6,-0.5,0) }}
    },
    // 2: multi_disconnect_fast
    {
        "multi_disconnect_fast", 20.0, 10.0,
        {{ Vector(100,60,10), Vector(100,60,10), Vector(100,60,10),
           Vector(100,60,10), Vector(100,60,10) }},
        {{ Vector(8,5,0), Vector(-8,5,0), Vector(0,0.5,0),
           Vector(0,-8,0), Vector(0.5,0,0) }}
    },
    // 3: cluster_split_2
    {
        "cluster_split_2", 25.0, 10.0,
        {{ Vector(100,62,10), Vector(100,58,10), Vector(100,60,10),
           Vector(100,62,10), Vector(100,58,10) }},
        {{ Vector(5,3,0), Vector(5,1,0), Vector(0,0,0),
           Vector(-5,-1,0), Vector(-5,-3,0) }}
    },
    // 4: relay_failure
    {
        "relay_failure", 20.0, 10.0,
        {{ Vector(70,80,10), Vector(70,40,10), Vector(100,60,10),
           Vector(130,80,10), Vector(130,40,10) }},
        {{ Vector(0,0,0), Vector(0,0,0), Vector(8,0,0),
           Vector(0,0,0), Vector(0,0,0) }}
    }
}};

// ── 전역 상태 ─────────────────────────────────────────────────────────────────
struct LinkStats
{
    double rssi_sum  = 0.0;
    int    rssi_cnt  = 0;
};

static LinkStats                        g_stats[NUM_UAVS][NUM_UAVS];
static std::map<Mac48Address, uint32_t> g_macToNode;
static NodeContainer                    g_nodes;
static std::ofstream                    g_out;
static std::string                      g_scenarioId;

// ── RSSI 기반 PLR 파생 (Python _direct_link 로직과 동일 경향) ─────────────────
static double ComputePlr(double rssi_dbm)
{
    // healthy zone: RSSI > -78 → PLR < 5%
    if (rssi_dbm >= -78.0)
    {
        double x = (-78.0 - rssi_dbm);
        return std::max(0.5, 1.5 + x * 0.5);
    }
    // degraded zone: -85 <= RSSI < -78 → PLR 5%~20%
    if (rssi_dbm >= -85.0)
    {
        double x = (-78.0 - rssi_dbm);  // 0..7
        return 5.0 + x * (15.0 / 7.0);
    }
    // disconnected zone: RSSI < -85 → PLR 20%~85%
    double x = (-85.0 - rssi_dbm);      // 0..∞
    return std::min(85.0, 20.0 + x * 5.0);
}

// ── MAC → 노드 ID 매핑 빌드 ───────────────────────────────────────────────────
static void BuildMacMap()
{
    for (uint32_t i = 0; i < NUM_UAVS; ++i)
    {
        Ptr<WifiNetDevice> dev =
            DynamicCast<WifiNetDevice>(g_nodes.Get(i)->GetDevice(0));
        if (dev)
        {
            Mac48Address mac =
                Mac48Address::ConvertFrom(dev->GetAddress());
            g_macToNode[mac] = i;
        }
    }
}

// ── MonitorSniffRx 콜백 (RSSI 수집) ──────────────────────────────────────────
static void OnMonitorSniffRx(std::string     context,
                              Ptr<const Packet> packet,
                              uint16_t          channelFreqMhz,
                              WifiTxVector      txVector,
                              MpduInfo          aMpdu,
                              SignalNoiseDbm    signalNoise,
                              uint16_t          staId)
{
    uint32_t rxNode = 0;
    if (std::sscanf(context.c_str(), "/NodeList/%u/", &rxNode) != 1) return;

    Ptr<Packet> p = packet->Copy();
    WifiMacHeader hdr;
    if (p->GetSize() < hdr.GetSerializedSize()) return;
    p->RemoveHeader(hdr);

    auto it = g_macToNode.find(hdr.GetAddr2());
    if (it == g_macToNode.end()) return;
    uint32_t txNode = it->second;
    if (txNode == rxNode || txNode >= NUM_UAVS || rxNode >= NUM_UAVS) return;

    g_stats[txNode][rxNode].rssi_sum += signalNoise.signal;
    g_stats[txNode][rxNode].rssi_cnt++;
}

// ── 주기적 CSV 로깅 ───────────────────────────────────────────────────────────
static void LogInterval()
{
    double now = Simulator::Now().GetSeconds();

    for (uint32_t src = 0; src < NUM_UAVS; ++src)
    {
        for (uint32_t dst = src + 1; dst < NUM_UAVS; ++dst)
        {
            // 양방향 RSSI 합산 평균
            double rssi_sum = g_stats[src][dst].rssi_sum
                            + g_stats[dst][src].rssi_sum;
            int    rssi_cnt = g_stats[src][dst].rssi_cnt
                            + g_stats[dst][src].rssi_cnt;
            double rssi = (rssi_cnt > 0) ? rssi_sum / rssi_cnt : NOISE_DBM;

            // PLR: RSSI 기반 파생
            double plr = ComputePlr(rssi);

            // 거리 (실제 위치 기반)
            Vector ps = g_nodes.Get(src)->GetObject<MobilityModel>()->GetPosition();
            Vector pd = g_nodes.Get(dst)->GetObject<MobilityModel>()->GetPosition();
            double dist = CalculateDistance(ps, pd);

            // 파생 피처
            double snr        = rssi - NOISE_DBM;
            double rtt        = 8.0 + dist * 0.22;
            double throughput = std::max(1.2, 16.0 - dist * 0.23);

            // 링크 상태 (Python과 동일 임계값)
            std::string state;
            if (rssi >= HEALTHY_RSSI_MIN && plr <= HEALTHY_PLR_MAX)
                state = "healthy";
            else if (rssi >= DEGRADED_RSSI_MIN && plr <= DEGRADED_PLR_MAX)
                state = "degraded";
            else
                state = "disconnected";

            int hop_count = (state == "disconnected") ? 0 : 1;

            // 최적 relay: 간소화 (UAV2 기본값)
            int opt_relay = 2;

            g_out << std::fixed << std::setprecision(4)
                  << g_scenarioId << ","
                  << now          << ","
                  << src          << ","
                  << dst          << ","
                  << std::setprecision(3)
                  << dist         << ","
                  << hop_count    << ","
                  << 0            << ","
                  << rssi         << ","
                  << snr          << ","
                  << plr          << ","
                  << rtt          << ","
                  << throughput   << ","
                  << state        << ","
                  << opt_relay    << "\n";

            // 통계 초기화
            g_stats[src][dst] = LinkStats{};
            g_stats[dst][src] = LinkStats{};
        }
    }

    double next = now + TIME_STEP;
    if (next < Simulator::GetMaximumSimulationTime().GetSeconds())
        Simulator::Schedule(Seconds(TIME_STEP), &LogInterval);
}

// ── 메인 ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    uint32_t scenarioIdx = 0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("scenario", "Scenario index 0-4", scenarioIdx);
    cmd.Parse(argc, argv);

    if (scenarioIdx >= SCENARIOS.size())
    {
        NS_ABORT_MSG("Scenario index must be 0-4. Got: " << scenarioIdx);
    }

    const ScenarioConfig& sc = SCENARIOS[scenarioIdx];
    g_scenarioId = sc.id;

    std::cout << "Running scenario [" << scenarioIdx << "]: "
              << sc.id << "  duration=" << sc.duration_s << "s\n";

    // ── 노드 생성 ──────────────────────────────────────────────────────────
    g_nodes.Create(NUM_UAVS);

    // ── WiFi 설정 (802.11g ad-hoc) ─────────────────────────────────────────
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",   StringValue("ErpOfdmRate24Mbps"),
                                 "ControlMode",StringValue("ErpOfdmRate6Mbps"));

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                   "Exponent",     DoubleValue(2.7),
                                   "ReferenceLoss",DoubleValue(46.6777));

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(16.0));
    wifiPhy.Set("TxPowerEnd",   DoubleValue(16.0));
    wifiPhy.Set("RxGain",       DoubleValue(0.0));
    wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, g_nodes);

    // ── 이동 모델 설정 ──────────────────────────────────────────────────────
    MobilityHelper mobility;
    auto posAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < NUM_UAVS; ++i)
        posAlloc->Add(sc.init_pos[i]);
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(g_nodes);

    for (uint32_t i = 0; i < NUM_UAVS; ++i)
    {
        auto model = g_nodes.Get(i)
                          ->GetObject<ConstantVelocityMobilityModel>();
        model->SetVelocity(sc.velocity[i]);
    }

    // ── 인터넷 스택 + OLSR (HELLO/TC 메시지로 데이터 프레임 지속 생성) ────
    OlsrHelper olsr;
    Ipv4StaticRoutingHelper staticRouting;
    Ipv4ListRoutingHelper listRouting;
    listRouting.Add(olsr, 10);
    listRouting.Add(staticRouting, 0);

    InternetStackHelper internet;
    internet.SetRoutingHelper(listRouting);
    internet.Install(g_nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(devices);

    // ── MAC 주소 맵 빌드 ────────────────────────────────────────────────────
    BuildMacMap();

    // ── MonitorSniffRx 트레이스 연결 ────────────────────────────────────────
    for (uint32_t i = 0; i < NUM_UAVS; ++i)
    {
        std::string path = "/NodeList/" + std::to_string(i)
            + "/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx";
        Config::Connect(path, MakeCallback(&OnMonitorSniffRx));
    }

    // ── 출력 파일 헤더 ──────────────────────────────────────────────────────
    std::string filename = "uav_ns3_" + sc.id + ".csv";
    g_out.open(filename);
    g_out << "scenario_id,time_s,src_uav,dst_uav,distance_m,"
          << "hop_count,blocked_building_count,"
          << "rssi_dbm_est,snr_db_est,plr_pct_est,rtt_ms_est,throughput_mbps_est,"
          << "link_state,optimal_relay_uav\n";

    // ── 주기적 로깅 시작 ────────────────────────────────────────────────────
    Simulator::Schedule(Seconds(TIME_STEP), &LogInterval);

    // ── 시뮬레이션 실행 ─────────────────────────────────────────────────────
    Simulator::Stop(Seconds(sc.duration_s));
    Simulator::Run();

    g_out.close();
    Simulator::Destroy();

    std::cout << "Done. Output: " << filename << "\n";
    return 0;
}
