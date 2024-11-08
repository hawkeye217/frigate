import ActivityIndicator from "@/components/indicators/activity-indicator";
import { useCameraPreviews } from "@/hooks/use-camera-previews";
import { useTimezone } from "@/hooks/use-date-utils";
import { useOverlayState } from "@/hooks/use-overlay-state";
import { usePersistence } from "@/hooks/use-persistence";
import { FrigateConfig } from "@/types/frigateConfig";
import { RecordingStartingPoint } from "@/types/record";
import {
  REVIEW_PADDING,
  ReviewFilter,
  ReviewSegment,
  ReviewSeverity,
  ReviewSummary,
  SegmentedReviewData,
} from "@/types/review";
import {
  getBeginningOfDayTimestamp,
  getEndOfDayTimestamp,
} from "@/utils/dateUtil";
import EventView from "@/views/events/EventView";
import { RecordingView } from "@/views/recording/RecordingView";
import axios from "axios";
import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import {
  parseAsFloat,
  useQueryState,
  useQueryStates,
  parseAsString,
  parseAsBoolean,
} from "nuqs";

export default function Events() {
  const { data: config } = useSWR<FrigateConfig>("config", {
    revalidateOnFocus: false,
  });
  const timezone = useTimezone(config);

  // recordings viewer

  const [severity, setSeverity] = useOverlayState<ReviewSeverity>(
    "severity",
    "alert",
  );

  const [showReviewed, setShowReviewed] = usePersistence("showReviewed", false);

  const [recording, setRecording] =
    useOverlayState<RecordingStartingPoint>("recording");

  const [reviewId] = useQueryState("id", {
    history: "push",
  });
  const [isLoading, setIsLoading] = useState(!!reviewId);
  const [isUpdatingFilter, setIsUpdatingFilter] = useState(false);

  const [filterParams, setFilterParams] = useQueryStates(
    {
      cameras: parseAsString,
      labels: parseAsString,
      zones: parseAsString,
      group: parseAsString,
      after: parseAsFloat,
      before: parseAsFloat,
      showAll: parseAsBoolean,
    },
    {
      history: "replace",
    },
  );

  const reviewFilter = useMemo<ReviewFilter>(
    () => ({
      cameras: filterParams.cameras?.split(","),
      labels: filterParams.labels?.split(","),
      zones: filterParams.zones?.split(","),
      group: filterParams.group === "default",
      after: filterParams.after ?? undefined,
      before: filterParams.before ?? undefined,
      showAll: filterParams.showAll ?? undefined,
    }),
    [filterParams],
  );

  const setReviewFilter = useCallback(
    (newFilter: ReviewFilter) => {
      setFilterParams((prev) => {
        const updatedParams = {
          cameras: newFilter.cameras?.join(",") ?? null,
          labels: newFilter.labels?.join(",") ?? null,
          zones: newFilter.zones?.join(",") ?? null,
          after: newFilter.after ?? null,
          before: newFilter.before ?? null,
          showAll: newFilter.showAll ?? null,
        };

        // Handle group logic
        if (
          config &&
          newFilter.showAll &&
          prev.group &&
          prev.group !== "default"
        ) {
          const groupCameras = config.camera_groups[prev.group]?.cameras;

          // Check if the group has only the "birdseye" camera
          const isBirdseyeOnly =
            groupCameras &&
            groupCameras.length === 1 &&
            groupCameras[0] === "birdseye";

          if (groupCameras && !isBirdseyeOnly) {
            updatedParams.cameras = groupCameras.join(",");
          }
        }

        return updatedParams;
      });
    },
    [config, setFilterParams],
  );

  useEffect(() => {
    if (!reviewId) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    axios
      .get(`review/${reviewId}`)
      .then((resp) => {
        if (resp.status == 200 && resp.data) {
          const startTime = resp.data.start_time - REVIEW_PADDING;
          const date = new Date(startTime * 1000);

          setReviewFilter({
            after: getBeginningOfDayTimestamp(date),
            before: getEndOfDayTimestamp(date),
          });
          setRecording(
            {
              camera: resp.data.camera,
              startTime,
              severity: resp.data.severity,
            },
            true,
          );
        }
      })
      .catch(() => {})
      .finally(() => {
        setIsLoading(false);
      });
  }, [reviewId, setRecording, setReviewFilter]);

  const [startTime, setStartTime] = useState<number>();

  useEffect(() => {
    if (recording) {
      document.title = "Recordings - Frigate";
    } else {
      document.title = `Review - Frigate`;
    }
  }, [recording, severity]);

  const onUpdateFilter = useCallback(
    (newFilterOrUpdater: React.SetStateAction<Partial<ReviewFilter>>) => {
      const currentFilter: ReviewFilter = reviewFilter ?? {};
      const newFilter =
        typeof newFilterOrUpdater === "function"
          ? newFilterOrUpdater(currentFilter)
          : newFilterOrUpdater;

      setReviewFilter(newFilter);

      // update recording start time if filter
      // was changed on recording page
      if (recording != undefined && newFilter.after != undefined) {
        setRecording({ ...recording, startTime: newFilter.after }, true);
      }
      // Use setTimeout to ensure state updates have been processed
      setTimeout(() => setIsUpdatingFilter(false), 0);
    },
    [recording, setRecording, setReviewFilter, reviewFilter],
  );

  // review paging

  const [beforeTs, setBeforeTs] = useState(Math.ceil(Date.now() / 1000));
  const last24Hours = useMemo(() => {
    return { before: beforeTs, after: getHoursAgo(24) };
  }, [beforeTs]);
  const selectedTimeRange = useMemo(() => {
    if (!filterParams.after) {
      return last24Hours;
    }
    return {
      before: Math.ceil(filterParams.before ?? Date.now() / 1000),
      after: Math.floor(filterParams.after),
    };
  }, [filterParams.after, filterParams.before, last24Hours]);

  // we want to update the items whenever the severity changes
  useEffect(() => {
    if (recording) {
      return;
    }

    const now = Date.now() / 1000;

    if (now - beforeTs > 60) {
      setBeforeTs(now);
    }

    // only refresh when severity changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [severity]);

  const reviewSegmentFetcher = useCallback((key: Array<string> | string) => {
    const [path, params] = Array.isArray(key) ? key : [key, undefined];
    return axios.get(path, { params }).then((res) => res.data);
  }, []);

  const getKey = useCallback(() => {
    const params = {
      cameras: filterParams.cameras,
      labels: filterParams.labels,
      zones: filterParams.zones,
      reviewed: 1,
      before: selectedTimeRange.before || last24Hours.before,
      after: selectedTimeRange.after || last24Hours.after,
    };
    return ["review", params];
  }, [filterParams, selectedTimeRange, last24Hours]);

  const { data: reviews, mutate: updateSegments } = useSWR<ReviewSegment[]>(
    getKey,
    reviewSegmentFetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const reviewItems = useMemo<SegmentedReviewData>(() => {
    if (!reviews) {
      return undefined;
    }

    const all: ReviewSegment[] = [];
    const alerts: ReviewSegment[] = [];
    const detections: ReviewSegment[] = [];
    const motion: ReviewSegment[] = [];

    reviews?.forEach((segment) => {
      all.push(segment);

      switch (segment.severity) {
        case "alert":
          alerts.push(segment);
          break;
        case "detection":
          detections.push(segment);
          break;
        default:
          motion.push(segment);
          break;
      }
    });

    return {
      all: all,
      alert: alerts,
      detection: detections,
      significant_motion: motion,
    };
  }, [reviews]);

  const currentItems = useMemo(() => {
    if (!reviewItems || !severity) {
      return null;
    }

    let current;

    if (reviewFilter?.showAll) {
      current = reviewItems.all;
    } else {
      current = reviewItems[severity];
    }

    if (!current || current.length == 0) {
      return [];
    }

    if (!showReviewed) {
      return current.filter((seg) => !seg.has_been_reviewed);
    } else {
      return current;
    }
    // only refresh when severity or filter changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [severity, reviewFilter, showReviewed, reviewItems?.all.length]);

  // review summary

  const { data: reviewSummary, mutate: updateSummary } = useSWR<ReviewSummary>(
    [
      "review/summary",
      {
        ...Object.fromEntries(
          Object.entries(filterParams)
            .filter(([, value]) => value !== undefined)
            .map(([key, value]) =>
              Array.isArray(value) ? [key, value.join(",")] : [key, value],
            ),
        ),
        timezone: timezone,
      },
    ],
    {
      revalidateOnFocus: true,
      refreshInterval: 30000,
      revalidateOnReconnect: false,
    },
  );

  const reloadData = useCallback(() => {
    setBeforeTs(Date.now() / 1000);
    updateSummary();
  }, [updateSummary]);

  // preview videos
  const previewTimes = useMemo(() => {
    const startDate = new Date(selectedTimeRange.after * 1000);
    startDate.setUTCMinutes(0, 0, 0);

    const endDate = new Date(selectedTimeRange.before * 1000);
    endDate.setHours(endDate.getHours() + 1, 0, 0, 0);

    return {
      after: startDate.getTime() / 1000,
      before: endDate.getTime() / 1000,
    };
  }, [selectedTimeRange]);

  const allPreviews = useCameraPreviews(
    previewTimes ?? { after: 0, before: 0 },
    {
      fetchPreviews: previewTimes != undefined,
    },
  );

  // review status

  const markAllItemsAsReviewed = useCallback(
    async (currentItems: ReviewSegment[]) => {
      if (currentItems.length == 0) {
        return;
      }

      const severity = currentItems[0].severity;
      updateSegments(
        (data: ReviewSegment[] | undefined) => {
          if (!data) {
            return data;
          }

          const newData = [...data];

          newData.forEach((seg) => {
            if (seg.end_time && seg.severity == severity) {
              seg.has_been_reviewed = true;
            }
          });

          return newData;
        },
        { revalidate: false, populateCache: true },
      );

      const itemsToMarkReviewed = currentItems
        ?.filter((seg) => seg.end_time)
        ?.map((seg) => seg.id);

      if (itemsToMarkReviewed.length > 0) {
        await axios.post(`reviews/viewed`, {
          ids: itemsToMarkReviewed,
        });
        reloadData();
      }
    },
    [reloadData, updateSegments],
  );

  const markItemAsReviewed = useCallback(
    async (review: ReviewSegment) => {
      const resp = await axios.post(`reviews/viewed`, { ids: [review.id] });

      if (resp.status == 200) {
        updateSegments(
          (data: ReviewSegment[] | undefined) => {
            if (!data) {
              return data;
            }

            const reviewIndex = data.findIndex((item) => item.id == review.id);
            if (reviewIndex == -1) {
              return data;
            }

            const newData = [
              ...data.slice(0, reviewIndex),
              { ...data[reviewIndex], has_been_reviewed: true },
              ...data.slice(reviewIndex + 1),
            ];

            return newData;
          },
          { revalidate: false, populateCache: true },
        );

        updateSummary(
          (data: ReviewSummary | undefined) => {
            if (!data) {
              return data;
            }

            const day = new Date(review.start_time * 1000);
            const today = new Date();
            today.setHours(0, 0, 0, 0);

            let key;
            if (day.getTime() > today.getTime()) {
              key = "last24Hours";
            } else {
              key = `${day.getFullYear()}-${("0" + (day.getMonth() + 1)).slice(-2)}-${("0" + day.getDate()).slice(-2)}`;
            }

            if (!Object.keys(data).includes(key)) {
              return data;
            }

            const item = data[key];
            return {
              ...data,
              [key]: {
                ...item,
                reviewed_alert:
                  review.severity == "alert"
                    ? item.reviewed_alert + 1
                    : item.reviewed_alert,
                reviewed_detection:
                  review.severity == "detection"
                    ? item.reviewed_detection + 1
                    : item.reviewed_detection,
              },
            };
          },
          { revalidate: false, populateCache: true },
        );
      }
    },
    [updateSegments, updateSummary],
  );

  // selected items

  const selectedReviewData = useMemo(() => {
    if (!recording) {
      return undefined;
    }

    if (!config) {
      return undefined;
    }

    if (!reviews) {
      return undefined;
    }

    setStartTime(recording.startTime);
    const allCameras = reviewFilter?.cameras ?? Object.keys(config.cameras);

    return {
      camera: recording.camera,
      start_time: recording.startTime,
      allCameras: allCameras,
    };

    // previews will not update after item is selected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording, reviews]);

  if (!timezone || isLoading || (recording && isUpdatingFilter)) {
    return (
      <ActivityIndicator className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2" />
    );
  }

  if (recording) {
    if (selectedReviewData) {
      return (
        <RecordingView
          key={selectedTimeRange.before}
          startCamera={selectedReviewData.camera}
          startTime={selectedReviewData.start_time}
          allCameras={selectedReviewData.allCameras}
          reviewItems={reviews}
          reviewSummary={reviewSummary}
          allPreviews={allPreviews}
          timeRange={selectedTimeRange}
          filter={reviewFilter}
          updateFilter={onUpdateFilter}
        />
      );
    }
  }

  return (
    <EventView
      reviewItems={reviewItems}
      currentReviewItems={currentItems}
      reviewSummary={reviewSummary}
      relevantPreviews={allPreviews}
      timeRange={selectedTimeRange}
      filter={reviewFilter}
      severity={severity ?? "alert"}
      startTime={startTime}
      showReviewed={showReviewed ?? false}
      setShowReviewed={setShowReviewed}
      setSeverity={setSeverity}
      markItemAsReviewed={markItemAsReviewed}
      markAllItemsAsReviewed={markAllItemsAsReviewed}
      onOpenRecording={setRecording}
      pullLatestData={reloadData}
      updateFilter={onUpdateFilter}
    />
  );
}

function getHoursAgo(hours: number): number {
  const now = new Date();
  now.setHours(now.getHours() - hours);
  return Math.ceil(now.getTime() / 1000);
}
